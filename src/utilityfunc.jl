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
    vector::Union{Vector{Number}, Missing} # can be missing for ASC
end

# a coef times a vector is a coefvector
*(coef::Coef, vector::Vector{Float64}) = CoefVector(coef, vector)
# This is for the first addition of two coefvectors in a utility function
+(cv::CoefVector, cv2::CoefVector) = [cv, cv2]
# this is for additional additions
+(vector::Vector{CoefVector}, cv::CoefVector) = [vector..., cv]

# special handling for ASCs
+(asc::Coef, cv::CoefVector) = [CoefVector(asc, missing), cv]
+(asc::Coef, vector::Vector{CoefVector}) = [CoefVector(asc, missing), vector...]
