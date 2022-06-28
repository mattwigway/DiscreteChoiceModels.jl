module DiscreteChoiceModels

using OnlineStats, Tables, Printf, Statistics, LinearAlgebra, DataFrames, Dagger, ForwardDiff, FLoops, SplittablesBase,
    Distributions, HaltonSequences, Primes
import FunctionWrappers: FunctionWrapper

include("LogitModel.jl")
include("util.jl")
include("utilitymacro.jl")
include("draws.jl")
include("mnl.jl")
include("mixed_logit.jl")
include("compute_utility.jl")

export multinomial_logit, mixed_logit, @utility, @β, @dummy_code

end