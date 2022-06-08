#=
Estimation routines for a mixed logit model.
=#

using Infiltrator

struct MixedLogitModel <: LogitModel
    coefnames::Vector{Symbol}
    coefs::Vector{Float64}
    mixed_coefnames::Vector{Symbol}
    mixed_coefs::Vector{UnivariateDistribution}
    vcov::Matrix{Float64}
    init_ll::Float64
    final_ll::Float64
    realized_draws::Array{Float64, 3}
    drawtype::DrawType.T
    # TODO log likelihood at constants
end

function mixed_logit_log_likelihood(utility_functions, chosen_col, avail_cols, data, parameters::Vector{T},
        mixed_coefs, draws)::T where T
    # First, realize the draws using current distribution paramters
    realized_coefs = zeros(T, size(draws))

    for (cidx, mixed_coef) in enumerate(mixed_coefs)
        distr = mixed_coef(parameters)::UnivariateDistribution
        # Halton points are between 0 and 1. Calling quantile will convert them to points that are
        # distributed according to the distribution 
        realized_coefs[cidx,:,:] = quantile.(distr, draws[cidx,:,:])
    end

    # mixed_draws now contains appropriately distributed values given the current distributions
    R = rowtype(data)
    U = typeof(utility_functions)
    C = typeof(chosen_col)
    A = typeof(avail_cols)
    ll = rowwise_loglik(
        FunctionWrapper{T, Tuple{Int64, R, Vector{T}, Array{T, 3}, U, C, A}}(mixed_ll_row),
        data, parameters, realized_coefs, utility_functions, chosen_col, avail_cols)

    if isnan(ll)
        @infiltrate
    end
    ll
end

function mixed_ll_row(rownumber, row, params::Vector{T}, realized_coefs::Array{T, 3}, utility_functions,
        ::Val{chosen_col}, avail_cols)::T where {T <: Number, chosen_col}
    chosen = row[chosen_col]
    prob_sum = zero(T)

    for draw in 1:size(realized_coefs)[3]
        util_sum = zero(T)
        local chosen_exputil::T
        mixed_values = @view realized_coefs[:,rownumber,draw]  # TODO memory locality okay here?
        for (choiceidx, ufunc) in enumerate(utility_functions)
            exp_util = if isnothing(avail_cols) || extract_namedtuple_bool(row, @inbounds Val(avail_cols[choiceidx]))
                # choice is available, either implicitly or explicitly
                exp(ufunc(params, row, mixed_values))
            else
                zero(T)
            end

            if choiceidx == chosen
                chosen_exputil = exp_util
            end
            util_sum += exp_util
        end

        prob_sum += chosen_exputil / util_sum
    end

    return log(prob_sum / size(realized_coefs)[3])
end

function mixed_logit(
    utility,
    chosen,
    data::DataFrame; # to get this to work with dagger, we need to change the get_draws call to access columns differently
    availability::Union{Nothing, AbstractVector{<:Pair{<:Any, <:Any}}}=nothing,
    method=BFGS(),
    se=true,
    verbose=:no,
    iterations=1_000,
    draws=100
    )

    isempty(utility.mixed_coefs) && @warn "Mixed logit requested but no mixing requested"

    data, choice_col, avail_cols = prepare_data(data, chosen, utility.alt_numbers, availability)

    # realize draws
    # TODO don't realize all draws on one machine and pass them around, somehow attach draws to dagger rows. With a large
    # dataset this could get out of hand.
    # Maybe there's a way to do it where we have per-observation/per-level seeds for Monte Carlo, or
    # defined offsets for the Halton sequences
    realized_draws = get_draws(nrow(data), draws, map(x -> data[!, x], utility.mixed_levels), DrawType.Halton)

    row_type = rowtype(data)
    obj(p::AbstractVector{T}) where T = -mixed_logit_log_likelihood(
        FunctionWrapper{T, Tuple{Vector{T}, row_type, Vector{T}}}.(utility.utility_functions),
        Val(choice_col), avail_cols, data, p, utility.mixed_coefs, realized_draws)::T
    init_ll = -obj(utility.starting_values)

    @info "Simulated log-likelihood at starting values $(init_ll)"

    results = optimize(
        TwiceDifferentiable(obj, utility.starting_values, autodiff=:forward),
        copy(utility.starting_values),
        method,
        Optim.Options(show_trace=verbose == :medium || verbose == :high, extended_trace=verbose==:high, iterations=iterations)
    )

    if !Optim.converged(results)
        @error "Failed to converge!"
        #throw(ConvergenceException(Optim.iterations(results)))
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

    return MixedLogitModel(final_coefnames, final_coefs, utility.mixed_coefnames, map(x -> x(params), utility.mixed_coefs),
        vcov, init_ll, final_ll, realized_draws, DrawType.Halton)
end


ndraws(m::MixedLogitModel) = size(m.realized_draws)[2]
drawtype(m::MixedLogitModel) = m.drawtype

function Base.summary(res::MixedLogitModel)
    mcfadden = 1 - res.final_ll / res.init_ll
    header = """
Mixed logit model
Initial simulated log-likelhood (at starting values): $(res.init_ll)
Final simulated log-likelihood: $(res.final_ll)
$(ndraws(res)) $(drawtype(res)) draws
McFadden's pseudo-R2 (relative to starting values): $mcfadden
"""

    vc = vcov(res)
    # nan variance in fixed params
    if !all(filter(x->!isnan(x), diag(vc)) .≥ 0)
        @error "Some estimated variances are negative, not showing std. errors!"
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
