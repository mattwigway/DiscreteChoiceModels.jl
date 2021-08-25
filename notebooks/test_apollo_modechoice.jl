using CSV
using DataFrames
using DiscreteChoiceModels

data = CSV.read("../data/apollo_modeChoiceData.csv", DataFrame)
data = data[data.RP .== 1, :]

model = multinomial_logit(
    [  # define utility functions
        1 => Coef(:tt_car) * data.time_car + Coef(:cost) * data.cost_car,
        2 => Coef(:asc_bus) + Coef(:tt_bus) * data.time_bus +
            Coef(:access) * data.access_bus + Coef(:cost) * data.cost_bus,
        3 => Coef(:asc_air) + Coef(:tt_air) * data.time_air +
            Coef(:access) * data.access_air + Coef(:cost) * data.cost_air,
        4 => Coef(:asc_rail) + Coef(:tt_rail) * data.time_rail +
            Coef(:access) * data.access_rail + Coef(:cost) * data.cost_rail
    ],
    data.choice,
    availability=[
        1 => data.av_car .== 1,
        2 => data.av_bus .== 1,
        3 => data.av_air .== 1,
        4 => data.av_rail .== 1
    ]
)

println(DiscreteChoiceModels.summary(model))