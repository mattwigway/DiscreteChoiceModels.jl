# The Apollo mixed logit example on Swiss route choice data

using DiscreteChoiceModels, DataFrames, CSV, Distributions

get_data() = CSV.read("test/data/apollo_swissRouteChoiceData.csv", DataFrame)

function main(data)
    mixed_logit(
        @utility(begin
            1 ~ -βtt * tt1 + -βtc * tc1 + -βhw * hw1 + -βch * ch1
            2 ~ -βtt * tt2 + -βtc * tc2 + -βhw * hw2 + -βch * ch2

            βtt = LogNormal(-3, exp(-4.6052)), level=>ID
            βtc = LogNormal(-3, exp(-4.6052)), level=>ID
            βhw = LogNormal(-3, exp(-4.6052)), level=>ID
            βch = LogNormal(-3, exp(-4.6052)), level=>ID
            # βtt = LogNormal(-1.9834, exp(-0.7631405523804803)), level=>ID
            # βtc = LogNormal(-1.0246, exp(-0.01582)), level=>ID
            # βhw = LogNormal(-2.9339, exp(-0.2030)), level=>ID
            # βch = LogNormal(0.6234, exp(-0.1908)), level=>ID
        end),
        :choice,
        data,
        verbose=:medium,
        draws=500
    )
end

model = main(get_data())
println(summary(model))