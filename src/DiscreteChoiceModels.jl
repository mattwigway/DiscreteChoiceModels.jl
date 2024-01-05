module DiscreteChoiceModels

using OnlineStats, Tables, Printf, Statistics, LinearAlgebra, DataFrames, Dagger, ForwardDiff, FLoops, SplittablesBase,
    Distributions, HaltonSequences, Primes, Logging
import FunctionWrappers: FunctionWrapper
import DTables: DTable

include("LogitModel.jl")
include("util.jl")
include("utilitymacro.jl")
include("draws.jl")
include("random_coefficients.jl")
include("mnl.jl")
include("nl.jl")
include("mixed_logit.jl")
include("compute_utility.jl")

export multinomial_logit, mixed_logit, nested_logit, @utility, @β, @dummy_code

end