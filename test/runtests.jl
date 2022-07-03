# Run tests

using SafeTestsets, Test, DiscreteChoiceModels, MacroTools, OnlineStats, CSV, DataFrames, StatsBase

@testset "Mixed logit" begin
    include("mixed/mixed.jl")
end

@testset "Multinomial logit" begin
    include("mnl/mnl.jl")
end

@testset "Macros" begin
    include("macro.jl")
end