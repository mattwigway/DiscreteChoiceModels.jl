using Optim
using PrettyTables
using LinearAlgebra
using Tables

struct MultinomialLogitModel <: LogitModel
    coefnames::Vector{Symbol}
    coefs::Vector{Float64}
    vcov::Matrix{Float64}
    init_ll::Float64
    const_ll::Float64
    final_ll::Float64
end

extract_val(::Val{T}) where T = T
# why does this get wrapped in an extra type?
extract_val(::Type{Val{T}}) where T = T

"""
This extracts a particular column from a row of a table, in a type stable manner. For some reason,
when working with a DTable, row[col] is type unstable even if the column name is known at compile time.
However, row.col is stable. gettyped is a generated function that takes advantage of this. 
gettyped(row, Val(col), T) is rewritten to row.col::T, which is type stable.
"""
@generated function gettyped(row, ::Val{col}, T)::T where col
    quote
        row.$col::T
    end
end

@generated function get_availability(row, ::Val{avail_cols}, i)::Bool where avail_cols
    exprs = Expr[]

    for i2 in eachindex(avail_cols)
        push!(exprs, quote
            if i == $i2
                return row.$(avail_cols[i2])::Bool
            end
        end)
    end

    push!(exprs, :(error("Invalid choice index!")))

    Expr(:block, exprs...)
end

#=
This directly calculate the loglikelihood for a single row, working in logarithmic space throughout to avoid overflows
    (h/t Sam Zhang).

Much work has gone into optimizing this to have zero allocations. Key optimizations:
- chosen_col is passed as a Val, so the column name is dispatched on. This means that the compiler knows which
  column will be used for chosen at compile time, and since it knows column types (from the type of NamedTuple row)
  can infer that chosen will always be an Int64 and avoid allocations
- Similarly, a barrier function with a val argument is used to prevent type instability when reading availability columns.
- params is received as a Vector{T} not an AbstractVector{T} - for some reason this saves two allocations
- FunctionWrappers are used to indicate to the compiler that ufunc will always return the same type (tested, the wrapper is necessary)

- Note that this was tested from a script calling multinomial_logit not inside a function, some of these optimizations may not be
  necessary inside a function.
=#
function mnl_ll_row(row, params::Vector{T}, utility_functions, ::Val{chosen_col}, ::Val{avail_cols})::T where {T <: Number, chosen_col, avail_cols}
    chosen = gettyped(row, Val(chosen_col), Int64)

    logsum::ImmutableLogSumExp{T} = ImmutableLogSumExp(T)

    chosen_util::T = zero(T)
    found_chosen = false
    for (choiceidx, ufunc) in enumerate(utility_functions)
        avail = isnothing(avail_cols) || get_availability(row, Val(avail_cols), choiceidx)

        if avail
            util::T = ufunc(params, row, nothing)
            logsum = update(logsum, util)

            if chosen == choiceidx
                found_chosen = true
                chosen_util = util
            end
        end
    end

    if !found_chosen
        error("Chosen value not available. Row: $row")
    end

    # calculate log-probability directly, no numerical errors
    # the probability is e^V_{chosen} / \sum_i e^{V_i}, i, chosen ∈ available.
    # the log probability, then, is ln e^V_{chosen} - ln \sum_i e^{V_i} by log rules, which further
    # simplifies to V_{chosen} - ln \sum_i e^{V_i}. LogSumExp uses various tricks to
    # calculate the logsum directly without overflow, and then we just subtract from the chosen utility.
    chosen_util - value(logsum)
end

function multinomial_logit_log_likelihood(utility_functions, chosen_col, avail_cols, data, parameters::Vector{T}) where T
    @debug "objective called with params" parameters
    R = rowtype(data)
    U = typeof(utility_functions)
    C = typeof(chosen_col)
    A = typeof(avail_cols)
    rowwise_loglik(
        FunctionWrapper{T, Tuple{R, Vector{T}, U, C, A}}(mnl_ll_row),
        data, parameters, utility_functions, chosen_col, avail_cols)
end

function multinomial_logit(
    utility,
    chosen,
    data;
    availability::Union{Nothing, AbstractVector{<:Pair{<:Any, <:Any}}}=nothing,
    method=BFGS(),
    se=true,
    verbose=false,
    iterations=1_000,
    logfile=nothing,
    include_ll_const=true,
    resume_from=nothing,
    resume_from_iteration=-1,
    allow_convergence_failure=false,
    optimization_options=()
    )

    # backwards-compatibility
    if verbose == :no
        verbose = false
    elseif verbose == :high || verbose == :medium
        verbose = true
    end

    if !isnothing(resume_from)
        # set starting values from logfile
        warm_start!(utility, resume_from, resume_from_iteration)
        @info "Resuming from $resume_from iteration $resume_from_iteration:" Dict(utility.coefnames .=> utility.starting_values)
    end

    isempty(utility.mixed_coefs) || error("Cannot have mixed coefs in multinomial logit model")
    
    check_utility(utility, data)

    data, choice_col, avail_cols = prepare_data(data, chosen, utility.alt_numbers, availability)
    row_type = rowtype(data)
    obj(p::AbstractVector{T}) where T = -multinomial_logit_log_likelihood(
        FunctionWrapper{T, Tuple{Vector{T}, row_type, Nothing}}.(utility.utility_functions),
        Val(choice_col), Val(avail_cols), data, p)::T
    init_ll = -obj(utility.starting_values)

    @info "Log-likelihood at starting values $(init_ll)"

    ll_const = if include_ll_const
        @info "Calculating log-likelihood at constants"
        util_const = constant_utility(utility)
        # Loglikelihood at constants for a mixed logit is just a multinomial logit
        # TODO this is actually much simpler - ll at constants is just predicting base rates
        const_model = #with_logger(NullLogger()) do
            multinomial_logit(util_const, chosen, data, availability=availability, method=method, se=false, include_ll_const=false, optimization_options=optimization_options)
        #end
        ll_const = loglikelihood(const_model)
        @info "Log-likelihood at constants: $(ll_const)"
        ll_const
    else
        NaN
    end

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
        Optim.Options(extended_trace=true, iterations=iterations, callback=state -> iteration_callback(logio, verbose, state);
            optimization_options...)
    )

    if !isnothing(logio)
        close(logio)
    end

    if !Optim.converged(results)
        # store results for investigation
        postmortem = "convergence-failure-$(uuid4()).jlobj"
        serialize(postmortem, results)
        @error "Model failed to converge after $(Optim.iterations(results)) iterations; serializing post-mortem results to $postmortem" results
        if !allow_convergence_failure
            throw(ConvergenceException(Optim.iterations(results)))
        end
    else
        @info "Optimization converged successfully after $(Optim.iterations(results)) iterations"
        if verbose
            @info "Optimization results" results
        end
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

    return MultinomialLogitModel(final_coefnames, final_coefs, vcov, init_ll, ll_const, final_ll)
end

function Base.summary(res::MultinomialLogitModel)
    mcfadden = 1 - loglikelihood(res) / nullloglikelihood(res)
    header = """
Multinomial logit model
Initial log-likelhood (at starting values): $(res.init_ll)
Log-likelihood at constants: $(nullloglikelihood(res))
Final log-likelihood: $(loglikelihood(res))
McFadden's pseudo-R2 (relative to constants): $mcfadden
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

    return header * table
end

multinomial_logit(::NamedTuple) = error("Not enough arguments. Make sure arguments to @utility are enclosed in parens")