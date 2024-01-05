# Replicate the Biogeme Swissmetro example
# https://biogeme.epfl.ch/examples/swissmetro/01logit.html

@testitem "Biogeme swissmetro nested" begin
    using CSV
    using DataFrames
    using DiscreteChoiceModels
    using Test
    using StatsBase

    data = CSV.File(joinpath(dirname(Base.source_path()), "../data/biogeme_swissmetro.dat"), delim='\t') |> DataFrame
    data = data[in.(data.PURPOSE, [Set([1, 3])]) .& (data.CHOICE .!= 0), :]

    @test nrow(data) == 6768

    data.avtr = (data.TRAIN_AV .== 1) .& (data.SP .!= 0)
    data.avsm = data.SM_AV .== 1
    data.avcar = (data.CAR_AV .== 1) .& (data.SP .!= 0)
    data.avex = data.avcar .|| data.avtr

    model = nested_logit(
        @utility(begin
            1 ~ αtrain + βtravel_time * TRAIN_TT / 100 + βcost * (TRAIN_CO * (GA == 0)) / 100
            2 ~ αswissmetro + βtravel_time * SM_TT / 100 + βcost * SM_CO * (GA == 0) / 100
            3 ~ αcar + βtravel_time * CAR_TT / 100 + βcost * CAR_CO / 100
            "Existing" ~ 0 # the nest has no shared observed attributes

            αswissmetro = 0, fixed

            # nesting structure
            "Existing" => [1, 3]
        end),
        :CHOICE,
        data,
        availability=[
            1 => :avtr,
            2 => :avsm,
            3 => :avcar,
            # TODO how to handle no members are available?
            "Existing" => :avex
        ]
    )

    coefs = Dict(zip(coefnames(model), round.(coef(model), sigdigits=3)))
    ses = Dict(zip(coefnames(model), round.(stderror(model), sigdigits=3)))

    println(coefs)

    @test coefs[:αcar] ≈ -0.167 # biogeme says 0.155 which would round to 0.16, but it's actually 0.1546
    @test coefs[:αswissmetro] == 0  # fixed
    @test coefs[:αtrain] ≈ -0.512
    @test coefs[:βcost] ≈ -0.857
    @test coefs[:βtravel_time] ≈ -0.899
    @test coefs[:θExisting] ≈ 0.487

    # note: these SEs come from the larch example, because it uses regular SEs while Biogeme uses
    # robust SEs (not yet supported in DCM.jl)
    # https://larch.newman.me/v5.7.0/example/109-swissmetro-nl.html
    @test ses[:αcar] ≈ 0.0371
    # @test isnan(ses[:αswissmetro])  # fixed
    @test ses[:αtrain] ≈ 0.0452
    # these two are scaled differently in this example vs larch example
    @test ses[:βcost] ≈ 0.0463
    @test ses[:βtravel_time] ≈ 0.0570

    @test round(loglikelihood(model), digits=1) ≈ -5236.9
    @test round(model.init_ll, digits=3) ≈ -6964.663
end