# Nested logit model. Following the definitions/notation in Koppelman, F. S., & Bhat, C. (2006).
#   A Self Instructing Course in Mode Choice Modeling: Multinomial and Nested Logit Models.
#   http://www.caee.utexas.edu/prof/bhat/courses/lm_draft_060131final-060630.pdf

struct NestedLogitModel <: LogitModel
    coefnames::Vector{Symbol}
    coefs::Vector{Float64}
    vcov::Matrix{Float64}
    init_ll::Float64
    const_ll::Float64
    final_ll::Float64
    utility::Any
    availability::Any
end

"""
Compute the scaled expected maximum utility (θΓ in Koppelman and Bhat).

The expected maximum utility (logsum) Γ for a nest is

Γ = ln Σ exp(V / θ), where V is the systematic utility for each member of the nest, θ is the inclusive value term,
and Γ is the expected maximum utility.

This is scaled by the inclusive value parameter θ when added to the systematic utility of the nest, so this function
returns a value multiplied by that inclusive value parameter.

Because we treat nests and elemental alternatives the same, if you request a "nest" that is actually just an alternative,
we return a scaled emu of zero.

In the Ben-Akiva and Lerman book, they do not have a Γ; instead they have a μ. Γ = 1 / μ; with this change the
formulations are equivalent.
"""
function get_scaled_emu(row, params::Vector{T}, iv_param_indices, utility_functions, nests, avail_cols, nest) where T
    if nest ∉ nests
        # leaf node/raw alternative/degenerate nest
        return zero(T)
    end

    # This directly stores ln Σ exp(V / θ)
    emu = ImmutableLogSumExp(T)

    θ = params[iv_param_indices[nest]]

    for (choiceidx, ufunc, other_nest) in zip(1:length(utility_functions), utility_functions, nests)
        avail = isnothing(avail_cols) || gettyped(row, Val(avail_cols[choiceidx]), Bool)
        if avail && other_nest == nest
            util = ufunc(params, row, nothing) + get_scaled_emu(row, params, iv_param_indices, utility_functions, nests, avail_cols, choiceidx)
            util /= θ
            emu = update(emu, util)
        end
    end


    # per Koppelman and Bhat, you do multiply this logged value by the param
    # Note: this is going to be -Inf if there is a nest where nothing is available
    scaled_emu = value(emu) * θ

    isfinite(scaled_emu) || error("non-finite scaled expected maximum utility (θΓ) at nest $nest, Γ=$(ForwardDiff.value(value(emu))), θ=$(ForwardDiff.value(iv_param)), row $row")

    return scaled_emu
end

function nl_ll_row(row, params::Vector{T}, utility_functions, nests, iv_param_indices, ::Val{chosen_col}, avail_cols)::T where {T <: Number, chosen_col}
    chosen = row[chosen_col]
    return nl_logprob(row, params, utility_functions, nests, iv_param_indices, chosen, avail_cols)
end

function nl_logprob(row, params::Vector{T}, utility_functions, nests, iv_param_indices, chosen, avail_cols) where T
    # in a nested logit model, the probability is the product of the conditional probabilities of all levels
    logprob = zero(T)

    current = chosen
    while current != TOP_LEVEL_NEST
        nest = nests[current]

        θ = if nest != TOP_LEVEL_NEST
            params[iv_param_indices[nest]]
        else
            one(T)
        end

        logsum = ImmutableLogSumExp(T)

        # calculate the log-probability for this nest/alternative, conditional on the parent
        local chosen_util::T
        found_chosen = false
        for alt in 1:length(utility_functions)
            avail = isnothing(avail_cols) || gettyped(row, Val(avail_cols[alt]), Bool)
            if avail && nests[alt] == nest
                # this alternative/nest is in the same nest, calculate its utility
                util = utility_functions[alt](params, row, nothing) + get_scaled_emu(row, params, iv_param_indices, utility_functions, nests, avail_cols, alt)
                util /= θ
                logsum = update(logsum, util)
                if alt == current
                    found_chosen = true
                    chosen_util = util
                end
            end
        end

        found_chosen || error("Chosen value not available. Row: $row")

        cond_logprob = chosen_util - value(logsum)
        logprob += cond_logprob

        current = nest
    end

    return logprob
end

function nested_logit_log_likelihood(utility_functions, nests, iv_param_indices, chosen_col, avail_cols, data, parameters::Vector{T}) where T
    R = rowtype(data)
    N = typeof(nests)
    I = typeof(iv_param_indices)
    U = typeof(utility_functions)
    C = typeof(chosen_col)
    A = typeof(avail_cols)
    ll = rowwise_loglik(
        FunctionWrapper{T, Tuple{R, Vector{T}, U, N, I, C, A}}(nl_ll_row),
        data, parameters, utility_functions, nests, iv_param_indices, chosen_col, avail_cols)
    @debug "objective is $(ForwardDiff.value(ll)) with params" ForwardDiff.value.(parameters)
    return ll
end

function nested_logit(
    utility,
    chosen,
    data;
    availability::Union{Nothing, AbstractVector{<:Pair{<:Any, <:Any}}}=nothing,
    method=BFGS(),
    se=true,
    verbose=false,
    iterations=1_000,
    include_ll_const=true,
    allow_convergence_failure=false,
    logfile=nothing
    )

    # backwards-compatibility
    if verbose == :no
        verbose = false
    elseif verbose == :high || verbose == :medium
        verbose = true
    end

    isempty(utility.mixed_coefs) || error("Cannot have mixed coefs in nested logit model")
    !all(utility.nests .== TOP_LEVEL_NEST) || @warn "No nests in nested logit model, it is equivalent to multinomial logit but less computationally efficient"

    data, choice_col, avail_cols = prepare_data(data, chosen, utility.alt_numbers, availability)
    row_type = rowtype(data)
    obj(p::AbstractVector{T}) where T = -nested_logit_log_likelihood(
        FunctionWrapper{T, Tuple{Vector{T}, row_type, Nothing}}.(utility.utility_functions),
        utility.nests, utility.iv_param_indices,
        Val(choice_col), avail_cols, data, p)::T
    init_ll = -obj(utility.starting_values)

    @info "Log-likelihood at starting values $(init_ll)"

    logio = if !isnothing(logfile)
        open(logfile, "w")
    else
        nothing
    end

    if !isnothing(logio)
        write_log_header(logio, utility)
    end

    results = optimize(
        TwiceDifferentiable(obj, utility.starting_values, autodiff=:forward),
        copy(utility.starting_values),
        method,
        Optim.Options(extended_trace=true, iterations=iterations, callback=state -> iteration_callback(logio, verbose, state))
    )

    if !isnothing(logio)
        close(logio)
    end

    if !Optim.converged(results)
        if allow_convergence_failure
            @warn "Failed to converge!"
        else
            throw(ConvergenceException(Optim.iterations(results)))
        end
    else
        @info "Optimization converged successfully after $(Optim.iterations(results)) iterations"
    end

    @info """
    Using method $(Optim.summary(results)),
    $(Optim.f_calls(results)) function evaluations, $(Optim.g_calls(results)) gradient evaluations
    """

    final_ll = -Optim.minimum(results)
    params = Optim.minimizer(results)

    # put any fixed parameters back into the data
    final_coefnames = [utility.coefnames..., keys(utility.fixed_coefs)...]
    final_coefs = [params..., values(utility.fixed_coefs)...]

    if se
        @info "Calculating and inverting Hessian"

        # compute standard errors
        hess = ForwardDiff.hessian(obj, params)
        local inv_hess
        try
            inv_hess = inv(hess)
        catch e
            !(e isa LinearAlgebra.SingularException) && rethrow()
            @error "Hessian is singular. Not reporting standard errors, and you should probably be suspicious of point estimates."
            se = false
        end

        if se
            vcov = similar(inv_hess, length(final_coefs), length(final_coefs))
            vcov[:, :] .= convert(eltype(vcov), NaN)
            vcov[1:length(params), 1:length(params)] = inv_hess
        end
    end

    if !se
        vcov = fill(NaN, length(final_coefs), length(final_coefs))
    end

    # TODO this nests parameter is not what we ultimately want, it still uses alternative codes
    return NestedLogitModel(final_coefnames, final_coefs, vcov, init_ll, NaN, final_ll, utility, availability)
end

function Base.summary(res::NestedLogitModel)
    header = """
Nested logit model
Initial log-likelhood (at starting values): $(res.init_ll)
Final log-likelihood: $(loglikelihood(res))
"""

    vc = vcov(res)
    # nan variance in fixed params
    if !all(filter(x->!isnan(x), diag(vc)) .≥ 0)
        @error "Some estimated variances are negative, not showing std. errors! Your model is likely not identified."
        ses = diag(vc)
        pval = fill(NaN, length(ses))
        selab = "Var"
    else
        ses = sqrt.(diag(vc))
        pval = coef(res) ./ ses
        selab = "Std. Err."
    end

    data = hcat(
        coefnames(res),
        coef(res),
        ses,
        pval
    )

    table = pretty_table(String, data, header=["", "Coef", selab, "Z-stat"],
        header_crayon=crayon"yellow bold", formatters=ft_printf("%.5f", 2:4))

    nesting_structure = draw_nesting_structure(res.utility)

    return header * table * "\nNesting structure:\n" * nesting_structure
end

function get_nesting_structure(utility, _from)
    if _from ∉ utility.nests
        return nothing
    end

    nesting_structure = Vector{Any}()
    for (alt, idx) in pairs(utility.alt_numbers)
        if utility.nests[idx] == _from
            subnests = get_nesting_structure(utility, idx)
            if isnothing(subnests)
                push!(nesting_structure, alt)
            else
                push!(nesting_structure, alt => subnests)
            end
        end
    end

    return nesting_structure
end

function _draw_nesting_structure!(structure, indent_level, buff)
    for nest in structure
        for _ in 1:indent_level
            print(buff, "  ")
        end

        if nest isa Pair
            println(buff, "- $(nest[1])")
            _draw_nesting_structure!(nest[2], indent_level + 1, buff)
        else
            println(buff, "- $nest")
        end
    end
end

function draw_nesting_structure(utility)
    structure = get_nesting_structure(utility, TOP_LEVEL_NEST)
    buff = IOBuffer()
    _draw_nesting_structure!(structure, 0, buff)
    return String(take!(buff))
end

nested_logit(::NamedTuple) = error("Not enough arguments. Make sure arguments to @utility are enclosed in parens")