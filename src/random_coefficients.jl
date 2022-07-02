# not mutable, new RandomCoefficients should be constructed each time params change
struct RandomCoefficients{T <: AbstractDraws}
    draws::T
    distributions::Vector{UnivariateDistribution}
end

RandomCoefficients(draws::T, mixed_coefs, params) where T = RandomCoefficients{T}(draws, [c(params) for c in mixed_coefs])

# call this with both row and group index; it will figure out which is which
getcoefdraw(coefs::RandomCoefficients, coef_index, row_index, group_index, draw_index) = quantile(coefs.distributions[coef_index], getdraw(coefs.draws, coef_index, row_index, group_index, draw_index))

Base.length(r::RandomCoefficients) = length(r.distributions)
ndraws(r::RandomCoefficients) = ndraws(r.draws)