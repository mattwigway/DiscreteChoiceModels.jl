# Code for specifying utility functions
import Base: *, +

# represents a coefficient
# immutable, but only used in constructing models - not used in optimization
struct Coef
    name::String
    starting_value::Float64
end

Coef(name::String) = Coef(name, 0)
Coef(name::Symbol) = Coef(string(name))
Coef(name::Symbol, starting_value::Float64) = Coef(string(name), starting_value)

# represents a coef multiplied by a vector
# eventually, we'll make it so this can do fancy stuff like interactions,
# squared terms, etc., inline
struct CoefVector
    coef::Coef
    vector::Union{Vector{<:Number}, Missing} # can be missing for ASC
end

# like a coefvector, but with an index into the array of coefs
# rather than a name for the coef. used in optimization.
struct InternalCoefVector
    index::Int64
    vector::Union{Missing, Vector{Number}}
end


# a coef times a vector is a coefvector
*(coef::Coef, vector::Vector{<:Number}) = CoefVector(coef, vector)
# This is for the first addition of two coefvectors in a utility function
+(cv::CoefVector, cv2::CoefVector) = [cv, cv2]
# this is for additional additions
+(vector::Vector{CoefVector}, cv::CoefVector) = [vector..., cv]

# special handling for ASCs
+(asc::Coef, cv::CoefVector) = [CoefVector(asc, missing), cv]
+(asc::Coef, vector::Vector{CoefVector}) = [CoefVector(asc, missing), vector...]

# parse a utility function, and place the following variables into the local scope
# Optim.jl operates on parameter arrays, not dataframes, so everything is indexed in this function with integers
# coefs - vector of unique Coefs
# starting_values - vector of starting values, parallel to coefs
# indexed_util - Vector of utility functions for alternatives, consisting of InternalCoefVectors
# indexed_chosen - Vector{Int64} of which utility function corresponds to the chosen option for each individual
# avail_mat - BitMatrix with availability for each alternative for each chooser
macro parseutility()
    # accumulate all unique coefs (as they may appear in multiple utility functions)
    return quote
        unique_coefs_set = Set{Coef}()
        for (uname, util_func) in $(esc(:(utility)))
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
        $(esc(:(coefs))) = coefs
        $(esc(:(starting_values::Vector{Float64} = map(c -> c.starting_value, coefs))))

        # ossify order
        ordered_util = [$(esc(:(utility)))...]
        $(esc(:(indexed_utility))) = map(ordered_util) do (name, util_func)
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

        local_avail_mat = nothing
        if !isnothing($(esc(:(availability))))
            local_avail_mat = BitMatrix(undef, length($(esc(:(chosen)))), length($(esc(:(availability)))))
            for (name, avvec) in $(esc(:(availability)))
                index = findfirst(u -> u.first == name, ordered_util)
                @assert !isnothing(index)
                local_avail_mat[:, index] = avvec
            end
        end
        $(esc(:(avail_mat))) = local_avail_mat

        # inefficient, could cache if becomes slow
        $(esc(:(indexed_chosen))) = map(choice -> findfirst(util -> util.first == choice, $(esc(:(utility)))), $(esc(:(chosen))))
        if any(isnothing.($(esc(:(indexed_chosen)))))
            error("some choices not found in index (internal error)")
        end
    end
end