module DiscreteChoiceModels

using OnlineStats, Tables, Printf, Statistics, LinearAlgebra, DataFrames, Dagger, ForwardDiff

include("LogitModel.jl")
include("util.jl")
include("utilitymacro.jl")
include("mnl.jl")
include("compute_utility.jl")

export multinomial_logit, @utility, @β, @dummy_code

end