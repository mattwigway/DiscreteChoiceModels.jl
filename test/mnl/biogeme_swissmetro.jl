# Replicate the Biogeme Swissmetro example
# https://biogeme.epfl.ch/examples/swissmetro/01logit.html

@testset "Biogeme swissmetro" begin
    using CSV
    using DataFrames
    using DiscreteChoiceModels
    using Test
    using StatsBase

    data = CSV.read(joinpath(dirname(Base.source_path()), "../data/biogeme_swissmetro.dat"), DataFrame, delim='\t')
    data = data[in.(data.PURPOSE, [Set([1, 3])]) .& (data.CHOICE .!= 0), :]

    @test nrow(data) == 6768

    model = multinomial_logit(
        @utility(begin
            1 ~ :αtrain + :βtravel_time * TRAIN_TT / 100 + :βcost * (TRAIN_CO * (GA == 0)) / 100
            2 ~ :αswissmetro + :βtravel_time * SM_TT / 100 + :βcost * SM_CO * (GA == 0) / 100
            3 ~ :αcar + :βtravel_time * CAR_TT / 100 + :βcost * CAR_CO / 100

            :αswissmetro = 0f  # fix swissmetro ASC to zero 
        end),
        data.CHOICE,
        data,
        availability=[
            1 => (data.TRAIN_AV .== 1) .& (data.SP .!= 0),
            2 => data.SM_AV .== 1,
            3 => (data.CAR_AV .== 1) .& (data.SP .!= 0),
        ]
    )

    coefs = Dict(zip(coefnames(model), round.(coef(model), digits=2)))
    ses = Dict(zip(coefnames(model), round.(stderror(model), digits=4)))

    @test coefs[:αcar] ≈ -0.15 # biogeme says 0.155 which would round to 0.16, but it's actually 0.1546
    @test coefs[:αswissmetro] == 0  # fixed
    @test coefs[:αtrain] ≈ -0.70
    @test coefs[:βcost] ≈ -1.08
    @test coefs[:βtravel_time] ≈ -1.28

    @test ses[:αcar] ≈ 0.0432
    @test isnan(ses[:αswissmetro])  # fixed
    @test ses[:αtrain] ≈ 0.0549
    @test ses[:βcost] ≈ 0.0518
    @test ses[:βtravel_time] ≈ 0.0569
end