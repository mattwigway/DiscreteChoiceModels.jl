"""
Code to compute draws for a mixed logit model. Consists of functions that return nvar x ndraws matrices.
These seem like they should be transposed, but they will be accessed one draw at a time in estimating, and
Julia stores matrices in Fortran (column-major) order.
"""

"""
Draws objects define methods to get draws. Methods:

getdraw(Draws, coef_index, group_index, draw_index)

where coef_index is which random coefficient, group_index is which group
(individual or observation), and draw_index is which draw.

getdraw() returns a float between 0 and 1 exclusive, which can be passed to quantile to get draws with a particular distribution.

getdraw() is pure, and should be deterministic even on different machine types (to allow for distributed computation).
"""
abstract type AbstractDraws end

getdraw(d::AbstractDraws, ::Int64, ::Int64, ::Int64) = error("$(typeof(d)) does not have a getdraws() function implemented")

# Halton draws are very simple, no configurable parameters
struct HaltonDraws <: AbstractDraws
    sequences::Vector{Halton}
    n_draws::Int64
    intra::BitVector  # are they intra or inter-individual draws
end

HaltonDraws(n_columns::Integer, n_draws, intra) = HaltonDraws(collect(map(Halton ∘ prime, 1:n_columns)), n_draws, intra)

# match order from apollo: Individual/obs 1 has halton draws [1:ndraws], 2 has [ndraws + 1:2*ndraws]
function getdraw(draws::HaltonDraws, coef_index, row_index, group_index, draw_index)
    offset_rows = draws.intra[coef_index] ? row_index : group_index
    draws.sequences[coef_index][draws.n_draws * (offset_rows - 1) + draw_index]
end

ndraws(d::HaltonDraws) = d.n_draws