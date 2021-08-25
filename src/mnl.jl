using Optim
using PrettyTables
using ForwardDiff
using LinearAlgebra

struct MNLResult
    coefs::Dict{String, Float64}
    ses::Dict{String, Float64}
    init_ll::Float64
    final_ll::Float64
    # TODO log likelihood at constants
end

# like a coefvector, but with an index into the array of parameters
# rather than a name
struct InternalCoefVector
    index::Int64
    vector::Union{Missing, Vector{Number}}
end

function multinomial_logit_log_likelihood(indexed_utility, indexed_choice, avail_mat, params)
    # make the vector the same as the element type of params so ForwardDiff works
    thread_ll = zeros(eltype(params), Threads.nthreads())
    Threads.@threads for obs in 1:length(indexed_choice)
        # compute all utilities
        exp_utils = map(enumerate(indexed_utility)) do (i, u)
            if isnothing(avail_mat) || avail_mat[obs, i]
                util = convert(eltype(params), 0) # for ForwardDiff again
                # alt is available
                for cv in u
                    if ismissing(cv.vector)
                        # ASC, no vector
                        util += params[cv.index]
                    else
                        util += params[cv.index] * cv.vector[obs]
                    end
                end
                exputil = exp(util)
                # if !isfinite(exputil)
                #     error("Infinite value in exp(util)!")
                # end
                return exputil
            else
                # unavailable is util = -inf, exp(-inf) = 0
                return convert(eltype(params), 0)
            end
        end

        logprob = log(exp_utils[indexed_choice[obs]] / sum(exp_utils))
        thread_ll[Threads.threadid()] += logprob
    end
    total_ll = sum(thread_ll)
    return total_ll
end

function multinomial_logit(
    # Ideally would be Union{<:AbstractVector{CoefVector}, <:Number}}, but
    # typeof([1=>0, 2=>Coef(:base)]) = Pair{Int64, Any}
    utility::AbstractVector{<:Pair{<:Any, <:Any}},
    chosen::AbstractVector{<:Any};
    availability::Union{Nothing, AbstractVector{<:Pair{<:Any, <:AbstractVector{Bool}}}}=nothing
    )
    # accumulate all unique coefs (as they may appear in multiple utility functions)
    # TODO abstract this code out for NL, etc.
    unique_coefs_set = Set{Coef}()
    for (uname, util_func) in utility
        if typeof(util_func) <: AbstractVector{CoefVector}
            for coefvector in util_func
                push!(unique_coefs_set, coefvector.coef)
            end
        elseif typeof(util_func) <: CoefVector
            push!(unique_coefs_set, util_func.coef)
        elseif typeof(util_func) <: Coef
            push!(unique_coefs_set, util_func)
        elseif (typeof(util_func) <: Number) && (util_func == 0)
            # nothing to estimate
        else
            error("Utility $uname must contain coefficients or be zero")
        end
    end

    # TODO add check here to make sure that there aren't coefs with same name and different
    # starting values

    # make order concrete
    coefs = [unique_coefs_set...]
    starting_values::Vector{Float64} = map(c -> c.starting_value, coefs)

    # ossify order
    ordered_util = [utility...]
    indexed_utility = map(ordered_util) do (name, util_func)
        if typeof(util_func) <: AbstractVector{CoefVector}
            return map(util_func) do cv
                index = findfirst(c -> c.name == cv.coef.name, coefs)
                @assert !isnothing(index)
                return InternalCoefVector(index, cv.vector)
            end
        elseif typeof(util_func) <: CoefVector
            index = findfirst(c -> c.name == util_func.coef.name, coefs)
            @assert !isnothing(index)
            return [IndexedCoefVector(index, util_func.vector)]
        elseif typeof(util_func) <: Coef
            index = findfirst(c -> c.name == util_func.name, coefs)
            @assert !isnothing(index)
            return [IndexedCoefVector(index, missing)]
        elseif (typeof(util_func) <: Number) && (util_func == 0)
            return Vector{InternalCoefVector}()
        else
            # ugly, should be enforced by type system, see comment above
            error("Utility function $name must contain coefficients or be zero")
        end
    end

    avail_mat = nothing
    if !isnothing(availability)
        avail_mat = BitArray(undef, length(chosen), length(availability))
        for (name, avvec) in availability
            index = findfirst(u -> u.first == name, ordered_util)
            @assert !isnothing(index)
            avail_mat[:, index] = avvec
        end
    end

    # inefficient, could cache if becomes slow
    indexed_chosen = map(choice -> findfirst(util -> util.first == choice, utility), chosen)
    @assert !any(isnothing.(indexed_chosen))

    @info "Optimizing $(length(unique_coefs_set)) coefficients}"

    init_ll = multinomial_logit_log_likelihood(indexed_utility, indexed_chosen, avail_mat, starting_values)
    @info "Log-likelihood at starting values $(init_ll)"

    obj(p) = -multinomial_logit_log_likelihood(indexed_utility, indexed_chosen, avail_mat, p)
    results = optimize(
        obj,
        starting_values,
        BFGS();  # TODO don't hardwire method
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

    final_coefs = Dict(map(i -> (coefs[i].name => params[i]), 1:length(coefs)))
    final_ses = Dict(map(i -> (coefs[i].name => se[i]), 1:length(coefs)))

    return MNLResult(final_coefs, final_ses, init_ll, final_ll)
end

function summary(res::MNLResult)
    mcfadden = 1 - res.final_ll / res.init_ll
    header = """
Multinomial logit model
Initial log-likelhood (at starting values): $(res.init_ll)
Final log-likelihood: $(res.final_ll)
McFadden's pseudo-R2 (relative to starting values): $mcfadden
"""

    table_rows = collect(keys(res.coefs))
    data = hcat(
        table_rows,
        map(c -> res.coefs[c], table_rows),
        map(c -> res.ses[c], table_rows),
        map(c -> res.coefs[c] / res.ses[c], table_rows)
    )

    table = pretty_table(String, data, header=["", "Coef", "Std. Err.", "Z-stat"],
        header_crayon=crayon"yellow bold", formatters=ft_printf("%.5f", 2:4))

    return header * table
end