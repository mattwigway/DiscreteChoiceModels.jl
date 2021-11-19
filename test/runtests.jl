# Run tests

using SafeTestsets
using Test
using DiscreteChoiceModels
using MacroTools

@testset "Multinomial logit" begin
    include("mnl/mnl.jl")
end

@testset "Macros" begin
    include("macro.jl")
end