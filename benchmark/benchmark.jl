#=

Benchmark simple models

=#

using DiscreteChoiceModels
using CSV
using DataFrames
using BenchmarkTools
using Logging
using JuliaDB
using JuliaDBMeta

function benchmark_mnl()
    suite = BenchmarkGroup()

    data = CSV.read(joinpath(dirname(Base.source_path()), "../test/data/biogeme_swissmetro.dat"), DataFrame, delim='\t')
    data = data[in.(data.PURPOSE, [Set([1, 3])]) .& (data.CHOICE .!= 0), :]

    data.avtr = (data.TRAIN_AV .== 1) .& (data.SP .!= 0)
    data.avsm = data.SM_AV .== 1
    data.avcar = (data.CAR_AV .== 1) .& (data.SP .!= 0)

    suite["Biogeme example (DataFrame)"] = @benchmarkable multinomial_logit(
        @utility(begin
            1 ~ αtrain + βtravel_time * TRAIN_TT / 100 + βcost * (TRAIN_CO * (GA == 0)) / 100
            2 ~ αswissmetro + βtravel_time * SM_TT / 100 + βcost * SM_CO * (GA == 0) / 100
            3 ~ αcar + βtravel_time * CAR_TT / 100 + βcost * CAR_CO / 100

            αswissmetro = 0, fixed  # fix swissmetro ASC to zero 
        end),
        :CHOICE,
        $data,
        availability=[
            1 => :avtr,
            2 => :avsm,
            3 => :avcar,
        ]
    ) samples=100

    return suite
end

function benchmark_mnl_juliadb()
    # TODO this should benchmark across multiple workers
    suite = BenchmarkGroup()

    data = loadtable(joinpath(dirname(Base.source_path()), "../test/data/biogeme_swissmetro.dat"), delim='\t', distributed=true)

    # filter to wanted cases
    data = filter(data) do row
        ((row.PURPOSE == 1) | (row.PURPOSE == 3)) && row.CHOICE != 0
    end

    data = @transform data (
        avtr=(:TRAIN_AV == 1) && (:SP != 0),
        avsm=:SM_AV == 1,
        avcar=(:CAR_AV == 1) && (:SP != 0)
    )

    suite["Biogeme example (JuliaDB)"] = @benchmarkable multinomial_logit(
        @utility(begin
            1 ~ αtrain + βtravel_time * TRAIN_TT / 100 + βcost * (TRAIN_CO * (GA == 0)) / 100
            2 ~ αswissmetro + βtravel_time * SM_TT / 100 + βcost * SM_CO * (GA == 0) / 100
            3 ~ αcar + βtravel_time * CAR_TT / 100 + βcost * CAR_CO / 100

            αswissmetro = 0, fixed  # fix swissmetro ASC to zero 
        end),
        :CHOICE,
        $data,
        availability=[
            1 => :avtr,
            2 => :avsm,
            3 => :avcar,
        ]
    )
end


function main()
    Logging.disable_logging(Logging.Info)
    suite = BenchmarkGroup()
    suite["MNL_DataFrame"] = benchmark_mnl()
    suite["MNL_JuliaDB"] = benchmark_mnl_juliadb()
    tune!(suite)
    results = run(suite)
    # use median runtime as not skewed by initial compilation
    println(median(results))
end

main()