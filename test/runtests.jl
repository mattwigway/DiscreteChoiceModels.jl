# Run tests

using SafeTestsets
using Test
using DiscreteChoiceModels

@testset "Multinomial logit" begin
    include("mnl/mnl.jl")
end