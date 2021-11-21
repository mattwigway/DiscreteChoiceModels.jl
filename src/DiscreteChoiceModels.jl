module DiscreteChoiceModels

using OnlineStats, Tables, JuliaDB, JuliaDBMeta, Printf, Statistics, LinearAlgebra, DataFrames
import FunctionWrappers: FunctionWrapper

include("LogitModel.jl")
include("util.jl")
include("utilitymacro.jl")
include("mnl.jl")
include("compute_utility.jl")

export multinomial_logit, @utility, @β, @dummy_code

end