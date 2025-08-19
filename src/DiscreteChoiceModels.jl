module DiscreteChoiceModels

using OnlineStats, Tables, Printf, Statistics, LinearAlgebra, DataFrames, Dagger, ForwardDiff, FLoops, SplittablesBase,
    Distributions, HaltonSequences, Primes, Logging
import FunctionWrappers: FunctionWrapper
import ThreadsX, Dates
import DTables: DTable
import CSV
import Serialization: serialize
import UUIDs: uuid4

include("LogitModel.jl")
include("util.jl")
include("utilitymacro.jl")
include("draws.jl")
include("random_coefficients.jl")
include("mnl.jl")
include("nl.jl")
include("mixed_logit.jl")
include("compute_utility.jl")
include("check_utility.jl")
include("logging.jl")
include("foreachchoice.jl")

export multinomial_logit, mixed_logit, nested_logit, @utility, @β, @dummy_code, @foreachchoice

end