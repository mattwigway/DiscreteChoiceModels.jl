using Optim
using PrettyTables
using ForwardDiff
using LinearAlgebra
using Tables

struct MultinomialLogitModel <: LogitModel
    coefnames::Vector{Symbol}
    coefs::Vector{Float64}
    ses::Vector{Float64}  # ses missing for fixed params, in future add option to disable se estimation
    init_ll::Float64
    final_ll::Float64
    # TODO log likelihood at constants
end

function multinomial_logit_log_likelihood(utility_functions, numbered_chosen, data, availability, params)
    # make the vector the same as the element type of params so ForwardDiff works
    thread_ll = zeros(eltype(params), Threads.nthreads())
    Threads.@threads for (rowidx, choice, row) in collect(zip(1:length(numbered_chosen), numbered_chosen, Tables.rows(data)))
        exp_utils = map(enumerate(utility_functions)) do (choiceidx, ufunc)
            if isnothing(availability) || availability[rowidx, choiceidx]
                return exp(ufunc(params, row))
            else
                # unavailable is util = -inf, exp(-inf) = 0
                return zero(eltype(params))
            end
        end

        logprob = log(exp_utils[choice] / sum(exp_utils))
        thread_ll[Threads.threadid()] += logprob
    end
    total_ll = sum(thread_ll)
    return total_ll
end

function multinomial_logit(
    utility,
    chosen::AbstractVector{<:Any},
    data;
    availability::Union{Nothing, AbstractVector{<:Pair{<:Any, <:AbstractVector{Bool}}}}=nothing,
    method=BFGS()
    )

    numbered_chosen = getkey.([utility.alt_numbers], chosen, [-1])
    any(numbered_chosen .== -1) && error("not all choices appear in utility functions")

    # convert availability to a matrix
    avail_mat = availability_to_matrix(availability, utility.alt_numbers)

    init_ll = multinomial_logit_log_likelihood(utility.utility_functions, numbered_chosen, data, avail_mat, utility.starting_values)
    @info "Log-likelihood at starting values $(init_ll)"

    obj(p) = -multinomial_logit_log_likelihood(utility.utility_functions, numbered_chosen, data, avail_mat, p)
    results = optimize(
        obj,
        copy(utility.starting_values),
        method;
        autodiff = :forward  # pure-Julia likelihood function, autodiff for gradient/Hessian
    )

    if !Optim.converged(results)
        error("Did not converge")
    end

    @info """
    Optimization converged successfully after $(Optim.iterations(results)) iterations
    Using method $(Optim.summary(results)),
    $(Optim.f_calls(results)) function evaluations, $(Optim.g_calls(results)) gradient evaluations
    """

    final_ll = -Optim.minimum(results)
    params = Optim.minimizer(results)

    @info "Calculating and inverting Hessian"

    # compute standard errors
    hess = ForwardDiff.hessian(obj, params)
    inv_hess = inv(hess)
    se = sqrt.(diag(inv_hess))

    # put any fixed parameters back into the data
    final_coefnames = [utility.coefnames..., keys(utility.fixed_coefs)...]
    final_coefs = [params..., values(utility.fixed_coefs)...]
    final_ses = [se..., fill(NaN64, length(utility.fixed_coefs))...]

    return MultinomialLogitModel(final_coefnames, final_coefs, final_ses, init_ll, final_ll)
end

function summary(res::MultinomialLogitModel)
    mcfadden = 1 - res.final_ll / res.init_ll
    header = """
Multinomial logit model
Initial log-likelhood (at starting values): $(res.init_ll)
Final log-likelihood: $(res.final_ll)
McFadden's pseudo-R2 (relative to starting values): $mcfadden
"""

    data = hcat(
        res.coefnames,
        res.coefs,
        res.ses,
        res.coefs ./ res.ses
    )

    table = pretty_table(String, data, header=["", "Coef", "Std. Err.", "Z-stat"],
        header_crayon=crayon"yellow bold", formatters=ft_printf("%.5f", 2:4))

    return header * table
end

multinomial_logit(NamedTuple) = error("Not enough arguments. Make sure arguments to @utility are enclosed in parens")