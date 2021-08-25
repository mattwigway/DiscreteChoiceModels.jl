using Parquet
using DataFrames
using DiscreteChoiceModels

data = DataFrame(read_parquet("../data/swissmetro.parquet"))
data = data[in.(data.PURPOSE, [Set([1, 3])]) .& (data.CHOICE .!= 0), :]

@assert nrow(data) == 6768

model = multinomial_logit(
    [  # define utility functions
        # 1 is train, ASC remains at zero
        1 => Coef(:train) + Coef(:travel_time) * (data.TRAIN_TT ./ 100) + Coef(:cost) * ((data.TRAIN_CO .* (data.GA .== 0)) ./ 100),
        # Swissmetro
        2 => Coef(:travel_time) * (data.SM_TT ./ 100) + Coef(:cost) * ((data.SM_CO .* (data.GA .== 0)) ./ 100),
        3 => Coef(:car) + Coef(:travel_time) * (data.CAR_TT ./ 100) + Coef(:cost) * (data.CAR_CO ./ 100)
    ],
    data.CHOICE,
    availability=[
        1 => (data.TRAIN_AV .== 1) .& (data.SP .!= 0),
        2 => data.SM_AV .== 1,
        3 => (data.CAR_AV .== 1) .& (data.SP .!= 0),
    ]
)

println(DiscreteChoiceModels.summary(model))