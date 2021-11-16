using Optim
using PrettyTables
using ForwardDiff
using LinearAlgebra
using Tables
using DataFrames

struct MultinomialLogitModel <: LogitModel
    coefnames::Vector{Symbol}
    coefs::Vector{Float64}
    vcov::Matrix{Float64}
    init_ll::Float64
    final_ll::Float64
    # TODO log likelihood at constants
end

function multinomial_logit_log_likelihood(utility_functions, chosen_col, avail_cols, data, parameters)
    return rowwise_loglik(data, parameters) do row, params
        exp_utils = map(enumerate(utility_functions)) do (choiceidx, ufunc)
            if isnothing(avail_cols) || row[avail_cols[choiceidx]]
                # choice is available, either implicitly or explicitly
                return exp(ufunc(params, row))
            else
                # unavailable is util = -inf, exp(-inf) = 0
                return zero(eltype(params))
            end
        end

        logprob = log(exp_utils[row[chosen_col]] / sum(exp_utils))
        return logprob
    end
end

function multinomial_logit(
    utility,
    chosen,
    data;
    availability::Union{Nothing, AbstractVector{<:Pair{<:Any, <:Any}}}=nothing,
    method=BFGS(),
    se=true
    )

    if data isa JuliaDB.AbstractIndexedTable
        check_perfect_prediction(data, chosen, [utility.columnnames...])
    end

    data, choice_col, avail_cols = prepare_data(data, chosen, utility.alt_numbers, availability)

    init_ll = multinomial_logit_log_likelihood(utility.utility_functions, choice_col, avail_cols, data, utility.starting_values)
    @info "Log-likelihood at starting values $(init_ll)"

    obj(p) = -multinomial_logit_log_likelihood(utility.utility_functions, choice_col, avail_cols, data, p)
    results = optimize(
        TwiceDifferentiable(obj, copy(utility.starting_values), autodiff=:forward),
        copy(utility.starting_values),
        method,
        Optim.Options(show_trace=true)
    )

    if !Optim.converged(results)
        throw(ConvergenceException(Optim.iterations(results)))
    end

    @info """
    Optimization converged successfully after $(Optim.iterations(results)) iterations
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
            @warn "Hessian is singular. Not reporting standard errors, and you should probably be suspicious of point estimates."
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

    return MultinomialLogitModel(final_coefnames, final_coefs, vcov, init_ll, final_ll)
end

function Base.summary(res::MultinomialLogitModel)
    mcfadden = 1 - res.final_ll / res.init_ll
    header = """
Multinomial logit model
Initial log-likelhood (at starting values): $(res.init_ll)
Final log-likelihood: $(res.final_ll)
McFadden's pseudo-R2 (relative to starting values): $mcfadden
"""

    data = hcat(
        coefnames(res),
        coef(res),
        stderror(res),
        coef(res) ./ stderror(res)
    )

    table = pretty_table(String, data, header=["", "Coef", "Std. Err.", "Z-stat"],
        header_crayon=crayon"yellow bold", formatters=ft_printf("%.5f", 2:4))

    return header * table
end

multinomial_logit(NamedTuple) = error("Not enough arguments. Make sure arguments to @utility are enclosed in parens")