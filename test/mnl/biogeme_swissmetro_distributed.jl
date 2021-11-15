# Replicate the Biogeme Swissmetro example
# https://biogeme.epfl.ch/examples/swissmetro/01logit.html

using Distributed
procs = addprocs(4)
Distributed.@everywhere begin
    using Pkg; Pkg.activate(@__DIR__)
    Pkg.instantiate(); Pkg.precompile()
    using JuliaDB, JuliaDBMeta, DiscreteChoiceModels
end

@testset "Biogeme swissmetro distributed" begin
    using JuliaDB, JuliaDBMeta
    using DiscreteChoiceModels
    using Test
    using StatsBase

    @test length(workers()) == 4

    data = loadtable(joinpath(dirname(Base.source_path()), "../data/biogeme_swissmetro.dat"),
        distributed=true, chunks=4, delim='\t')

    rechunk(data; chunks=4)
    @test length(data.chunks) == 4

    # filter to wanted cases
    data = filter(data) do row
        ((row.PURPOSE == 1) | (row.PURPOSE == 3)) && row.CHOICE != 0
    end

    @test length(data) == 6768

    data = @transform data (
        avtr=(:TRAIN_AV == 1) && (:SP != 0),
        avsm=:SM_AV == 1,
        avcar=(:CAR_AV == 1) && (:SP != 0)
    )

    model = multinomial_logit(
        @utility(begin
            1 ~ αtrain + βtravel_time * TRAIN_TT / 100 + βcost * (TRAIN_CO * (GA == 0)) / 100
            2 ~ αswissmetro + βtravel_time * SM_TT / 100 + βcost * SM_CO * (GA == 0) / 100
            3 ~ αcar + βtravel_time * CAR_TT / 100 + βcost * CAR_CO / 100

            αswissmetro = 0, fixed  # fix swissmetro ASC to zero 
        end),
        :CHOICE,
        data,
        availability=[
            1 => :avtr,
            2 => :avsm,
            3 => :avcar,
        ]
    )

    coefs = Dict(zip(coefnames(model), round.(coef(model), sigdigits=3)))
    ses = Dict(zip(coefnames(model), round.(stderror(model), sigdigits=3)))

    @test coefs[:αcar] ≈ -0.155 # biogeme says 0.155 which would round to 0.16, but it's actually 0.1546
    @test coefs[:αswissmetro] == 0  # fixed
    @test coefs[:αtrain] ≈ -0.701
    @test coefs[:βcost] ≈ -1.08
    @test coefs[:βtravel_time] ≈ -1.28

    @test ses[:αcar] ≈ 0.0432
    @test isnan(ses[:αswissmetro])  # fixed
    @test ses[:αtrain] ≈ 0.0549
    @test ses[:βcost] ≈ 0.0518
    @test ses[:βtravel_time] ≈ 0.0569

    idx = Dict(zip(coefnames(model), 1:length(coefs)))
    vcovmat = round.(vcov(model), sigdigits=3)
    @test vcovmat[idx[:αtrain], idx[:αcar]] ≈ 0.00138
    @test vcovmat[idx[:βcost], idx[:αcar]] ≈ 0.000485
    @test vcovmat[idx[:βcost], idx[:αtrain]] ≈ 8.22e-6
    @test vcovmat[idx[:βtravel_time], idx[:αcar]] ≈ -0.00144
    @test vcovmat[idx[:βtravel_time], idx[:αtrain]] ≈ -0.00225
    @test vcovmat[idx[:βtravel_time], idx[:βcost]] ≈ 0.000550

    @test all(isnan.(vcovmat[idx[:αswissmetro], :]))
    @test all(isnan.(vcovmat[:, idx[:αswissmetro]]))

    @test round(loglikelihood(model), digits=3) ≈ -5331.252
    @test round(model.init_ll, digits=3) ≈ -6964.663
end

# clean up
rmprocs(procs)