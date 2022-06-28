# Run tests

using SafeTestsets, Test, DiscreteChoiceModels, MacroTools, OnlineStats

@testset "Logsumexp" begin
    include("logsumexp.jl")
end

@testset "Multinomial logit" begin
    include("mnl/mnl.jl")
end

@testset "Macros" begin
    include("macro.jl")
end