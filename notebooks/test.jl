using CSV
using DataFrames
using DiscreteChoiceModels

data = CSV.read("../data/biogeme_swissmetro.dat", DataFrame, delim='\t')
data = data[in.(data.PURPOSE, [Set([1, 3])]) .& (data.CHOICE .!= 0), :]

@assert nrow(data) == 6768

model = multinomial_logit(
    @utility(begin
        1 ~ :αtrain + :βtravel_time * TRAIN_TT / 100 + :βcost * (TRAIN_CO * (GA == 0)) / 100
        2 ~ :αswissmetro + :βtravel_time * SM_TT / 100 + :βcost * SM_CO * (GA == 0) / 100
        3 ~ :αcar + :βtravel_time * CAR_TT / 100 + :βcost * CAR_CO / 100

        :αswissmetro = 0f  # fix swissmetro ASC to zero 
    end),
    data.CHOICE,
    data
    # data,
    # availability=[
    #     1 => (data.TRAIN_AV .== 1) .& (data.SP .!= 0),
    #     2 => data.SM_AV .== 1,
    #     3 => (data.CAR_AV .== 1) .& (data.SP .!= 0),
    # ]
)

println(DiscreteChoiceModels.summary(model))