"""
Code to compute draws for a mixed logit model. Consists of functions that return nvar x ndraws matrices.
These seem like they should be transposed, but they will be accessed one draw at a time in estimating, and
Julia stores matrices in Fortran (column-major) order.
"""

module DrawType @enum T Halton end

# Take draws per observation.
# "Usually, different draws are taken for each observation. This procedure maintains independence over
# decision makers of the simulated probabilities that enter [simulated log likelihood]"
#   - Train, 2009, Discrete Choice Models with Simulation, pp. 144f
function get_draws(nobs, ndraws, levels, type::DrawType.T)
    draws = zeros(Float64, length(levels), nobs, ndraws)

    # fill in the columns
    for (colidx, level) in enumerate(levels)
        seqlength = isnothing(level) ? nobs * ndraws : length(unique(level)) * ndraws
        raw_draws = if type == DrawType.Halton
            # Like HaltonPoint, use the sequence of primes as bases for columns
            # see also Bhat 2003 Simulation Estimation of Mixed Discrete Choice Models Using Randomized and Scrambled Halton Sequences
            # for concerns about collinearity
            Halton(prime(colidx))[1:seqlength]
        else
            error("unknown draw type $type (should not be possible, report at https://github.com/mattwigway/DiscreteChoiceModels.jl/issues")
        end

        if isnothing(level)
            # TODO ensure that ordering is coming out right, this means each obs gets sequential
            # halton draws I think
            draws[colidx,:,:] = reshape(raw_draws, nobs, ndraws)
        else
            # figure out how many draws we need for this level
            unique_levels = unique(level)
            # an index into the unique_levels array that matches each level
            levelidxs = collect(map(x -> findfirst(unique_levels .== x), level))

            for draw in 1:ndraws
                raw_start = (draw - 1) * length(unique_levels) + 1
                raw_end = raw_start + length(unique_levels) - 1
                # Each obs get sequential halton draws I think
                draws[colidx, :, draw] = raw_draws[raw_start:raw_end][levelidxs]
            end
        end
    end

    draws
end
