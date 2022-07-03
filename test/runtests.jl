# Run tests

using SafeTestsets, Test, DiscreteChoiceModels, MacroTools, OnlineStats

@testset "Multinomial logit" begin
    include("mnl/mnl.jl")
end

@testset "Macros" begin
    include("macro.jl")
end