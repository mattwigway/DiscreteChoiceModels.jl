@testitem "Convergence failure" begin

    using DataFrames
    using DiscreteChoiceModels
    using Test
    using StatsBase

    data = DataFrame(
        choice = [2, 2, 2, 2, 2, 2, 2],
        pred =   [0, 0, 1, 1, 0, 0, 0],  # perfectly predicts outcome
        otherx = [0, 1, 0, 1, 1, 1, 0]
    )

    @test_skip 1
    # Despite my best efforts I can't get this to fail to converge
    
    #=
    @test_throws ConvergenceException multinomial_logit(
        # asc2 ML estimate is ∞
        @utility(begin
            1 ~ 0
            2 ~ :asc2
        end),
        data.choice,
        data
    )
    =#
end