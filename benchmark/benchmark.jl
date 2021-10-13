#=

Benchmark simple models

=#

using DiscreteChoiceModels
using CSV
using DataFrames
using BenchmarkTools
using Logging

function benchmark_mnl()
    data = CSV.read(joinpath(dirname(Base.source_path()), "../test/data/biogeme_swissmetro.dat"), DataFrame, delim='\t')
    data = data[in.(data.PURPOSE, [Set([1, 3])]) .& (data.CHOICE .!= 0), :]
    suite = BenchmarkGroup()

    suite["Biogeme example"] = @benchmarkable multinomial_logit(
            @utility(begin
                1 ~ :αtrain + :βtravel_time * TRAIN_TT / 100 + :βcost * (TRAIN_CO * (GA == 0)) / 100
                2 ~ :αswissmetro + :βtravel_time * SM_TT / 100 + :βcost * SM_CO * (GA == 0) / 100
                3 ~ :αcar + :βtravel_time * CAR_TT / 100 + :βcost * CAR_CO / 100
    
                :αswissmetro = 0f  # fix swissmetro ASC to zero 
            end),
            ($data).CHOICE,
            $data,
            availability=[
                1 => (($data).TRAIN_AV .== 1) .& (($data).SP .!= 0),
                2 => ($data).SM_AV .== 1,
                3 => (($data).CAR_AV .== 1) .& (($data).SP .!= 0),
            ]
        ) samples=100

    return suite
end

function main()
    Logging.disable_logging(Logging.Info)
    suite = BenchmarkGroup()
    suite["MNL"] = benchmark_mnl()
    tune!(suite)
    results = run(suite)
    println(results)
end

main()