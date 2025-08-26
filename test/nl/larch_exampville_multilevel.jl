@testitem "Multilevel nesting (larch exampville)" begin
    # this reproduces this model, which has two levels of nesting: https://larch.driftless.xyz/v6.0/examples/exampville/201_exville_mode_choice.html
    import DataFrames: DataFrame, leftjoin!, rename!, nrow
    import CSV
    import DiscreteChoiceModels: nested_logit, @utility
    import StatsBase: coefnames, coef, stderror, loglikelihood

    data = CSV.read(joinpath(@__DIR__, "..", "data", "larch", "tour.csv"), DataFrame)
    persons = CSV.read(joinpath(@__DIR__, "..", "data", "larch", "person.csv"), DataFrame)
    hh = CSV.read(joinpath(@__DIR__, "..", "data", "larch", "hh.csv"), DataFrame)
    skims = CSV.read(joinpath(@__DIR__, "..", "data", "larch", "skims.csv"), DataFrame)

    leftjoin!(data, hh[!, [:HHID, :INCOME, :HOMETAZ]], on=:HHID)
    rename!(data, :HOMETAZ=>:OTAZ)
    leftjoin!(data, skims, on=[:OTAZ, :DTAZ])
    leftjoin!(data, persons[!, [:HHID, :PERSONID, :AGE]], on=[:HHID, :PERSONID])

    data = data[data.TOURPURP .== 1, :]

    @test nrow(data) == 7564

    data.LOGINCOME = log.(data.INCOME)

    data.DA_AVAIL = data.AGE .≥ 16
    data.SR_AVAIL .= true
    data.WALK_AVAIL = data.WALK_TIME .< 60
    data.BIKE_AVAIL = data.BIKE_TIME .< 60
    data.TRANSIT_AVAIL = data.TRANSIT_FARE .> 0

    data.CAR_AVAIL = data.DA_AVAIL .|| data.SR_AVAIL
    data.MOTOR_AVAIL = data.CAR_AVAIL .|| data.TRANSIT_AVAIL
    data.NONMOTOR_AVAIL = data.WALK_AVAIL .|| data.BIKE_AVAIL

    model = nested_logit(@utility(begin
        # base mode: drive alone
        1 ~ βInVehTime * AUTO_TIME + βCost * AUTO_COST
        2 ~ αSR + βInVehTime * AUTO_TIME + βCost * AUTO_COST * 0.5 + βLogIncomeSR * LOGINCOME
        3 ~ αWalk + βNonMotorTime * WALK_TIME + βLogIncomeWalk * LOGINCOME
        4 ~ αBike + βNonMotorTime * BIKE_TIME + βLogIncomeBike * LOGINCOME
        5 ~ αTransit + βInVehTime * TRANSIT_IVTT + βOutVehTime * TRANSIT_OVTT +
            βCost * TRANSIT_FARE + βLogIncomeTransit * LOGINCOME

        "Car" ~ 0
        "Car" => [1, 2]
        θCar = 0.25

        "Motor" ~ 0
        "Motor" => ["Car", 5]
        θMotor = 0.5

        "NonMotor" ~ 0
        "NonMotor" => [3, 4]
        θNonMotor = 0.5
    end),
        :TOURMODE,
        data,
        availability=[
            1 => :DA_AVAIL,
            2 => :SR_AVAIL,
            3 => :WALK_AVAIL,
            4 => :BIKE_AVAIL,
            5 => :TRANSIT_AVAIL,
            "Car" => :CAR_AVAIL,
            "Motor" => :MOTOR_AVAIL,
            "NonMotor" => :NONMOTOR_AVAIL 
        ]
    )

    println(summary(model))

    coefs = Dict(coefnames(model) .=> round.(coef(model), sigdigits=3))
    ses = Dict(coefnames(model) .=> round.(stderror(model), sigdigits=3))

    @test coefs[:αBike] ≈ -0.258
    @test ses[:αBike] ≈ 1.34
    @test coefs[:αSR] ≈ 1.42
    @test ses[:αSR] ≈ 1.00
    @test coefs[:αTransit] ≈ 6.75 # larch gets 6.75529 which would be 6.76 but close enougn
    @test ses[:αTransit] ≈ 2.06
    @test coefs[:αWalk] ≈ 8.62
    @test ses[:αWalk] ≈ 1.14
    @test coefs[:βCost] ≈ -0.176
    @test ses[:βCost] ≈ 0.120
    @test coefs[:βInVehTime] ≈ -0.124
    @test ses[:βInVehTime] ≈ 0.0292
    @test coefs[:βOutVehTime] ≈ -0.255
    @test ses[:βOutVehTime] ≈ 0.0646
    @test coefs[:βNonMotorTime] ≈ -0.266
    @test ses[:βNonMotorTime] ≈ 0.0163
    @test coefs[:βLogIncomeBike] ≈ -0.197
    @test ses[:βLogIncomeBike] ≈ 0.124
    @test coefs[:βLogIncomeSR] ≈ -0.194
    @test ses[:βLogIncomeSR] ≈ 0.136
    @test coefs[:βLogIncomeWalk] ≈ -0.523
    @test ses[:βLogIncomeWalk] ≈ 0.100
    @test coefs[:βLogIncomeTransit] ≈ -0.557
    @test ses[:βLogIncomeTransit] ≈ 0.169

    @test coefs[:θCar] ≈ 0.259
    @test ses[:θCar] ≈ 0.181
    @test coefs[:θMotor] ≈ 0.802
    @test ses[:θMotor] ≈ 0.201
    @test coefs[:θNonMotor] ≈ 0.854
    @test ses[:θNonMotor] ≈ 0.112

    @test round(loglikelihood(model), digits=2) ≈ -3493.04
end