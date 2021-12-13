#=

Benchmark simple models

=#

using DiscreteChoiceModels, CSV, DataFrames, BenchmarkTools, Logging, Dagger, Distributed, CodecZlib

const SAMPLES = 10

get_test_file_path(file) = joinpath(Base.source_dir(), "../test/data/", file)

function benchmark_mnl()
    data = CSV.read(get_test_file_path("biogeme_swissmetro.dat"), DataFrame, delim='\t')
    data = data[in.(data.PURPOSE, [Set([1, 3])]) .& (data.CHOICE .!= 0), :]

    data.avtr = (data.TRAIN_AV .== 1) .& (data.SP .!= 0)
    data.avsm = data.SM_AV .== 1
    data.avcar = (data.CAR_AV .== 1) .& (data.SP .!= 0)

    # TODO ensure compilation time for the macro is included!!!! for a fair comparison.
    @benchmarkable(
        multinomial_logit(
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
        ),
        samples=SAMPLES,
        seconds=1e6,
        evals=1
     )
end

function benchmark_mnl_dtable()
    data = DTable(Tables.rowtable(CSV.File(get_test_file_path("biogeme_swissmetro.dat"), delim='\t')), 2000)

    # filter to wanted cases
    data = filter(data) do row
        ((row.PURPOSE == 1) | (row.PURPOSE == 3)) && row.CHOICE != 0
    end

    data = map(data) do r
        merge((
            avtr=(r.TRAIN_AV == 1) && (r.SP != 0),
            avsm=r.SM_AV == 1,
            avcar=(r.CAR_AV == 1) && (r.SP != 0)
        ), pairs(r))
    end

    @benchmarkable(
        multinomial_logit(
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
        ),
        samples=SAMPLES,
        seconds=1e6,
        evals=1
    )
end

# benchmark the actual macroexpand, since it would be only compiled once during the other benchmarking
# https://discourse.julialang.org/t/benchmark-macro-with-benchmarktools/72723/2
function benchmark_mnl_macroexpand()
    @benchmarkable(@macroexpand(@utility(begin
                1 ~ αtrain + βtravel_time * TRAIN_TT / 100 + βcost * (TRAIN_CO * (GA == 0)) / 100
                2 ~ αswissmetro + βtravel_time * SM_TT / 100 + βcost * SM_CO * (GA == 0) / 100
                3 ~ αcar + βtravel_time * CAR_TT / 100 + βcost * CAR_CO / 100

                αswissmetro = 0, fixed  # fix swissmetro ASC to zero 
            end)),
            samples=SAMPLES,
            seconds=1e6,
            evals=1)
end

#########################################################
# NHTS Model                                            #
#########################################################

# convenience functions
# to save space in the repo, data are gzipped. helper function to read a
# gzipped csv "straight" into JuliaDB or DataFrames (via a temporary file)
function read_gzipped_csv(file, T; kwargs...)
    open(GzipDecompressorStream, file) do is
        mktemp() do path, os
            write(os, read(is))
            if T <: DTable
                DTable(CSV.File(path), 17500, kwargs...)
            elseif T <: DataFrame
                CSV.read(path, DataFrame, kwargs...)
            else
                error("Don't know how to read table into type $T")
            end
        end
    end
end

function benchmark_mnl_nhts_dtable()
    data = read_gzipped_csv(get_test_file_path("nhts/hhpub.csv.gz"), DTable)

    # topcode hhvehcnt
    data = map(r -> merge((hhveh_topcode=min(r.HHVEHCNT, 4),), pairs(r)), data)
    
    @benchmarkable(
        multinomial_logit(
            @utility(begin
                0 ~ α0
                1 ~ α1 + β1homeown * (HOMEOWN == 2) + @dummy_code(β1, HH_RACE, [2, 3, 4, 5, 6, 97]) + β1_hhsize * HHSIZE
                2 ~ α2 + β2homeown * (HOMEOWN == 2) + @dummy_code(β2, HH_RACE, [2, 3, 4, 5, 6, 97]) + β2_hhsize * HHSIZE
                3 ~ α3 + β3homeown * (HOMEOWN == 2) + @dummy_code(β3, HH_RACE, [2, 3, 4, 5, 6, 97]) + β3_hhsize * HHSIZE
                4 ~ α4plus + β4plushomeown * (HOMEOWN == 2) + @dummy_code(β4plus, HH_RACE, [2, 3, 4, 5, 6, 97]) + β4plus_hhsize * HHSIZE

                α0 = 0, fixed
            end),
            :hhveh_topcode,
            $data
        ),
        samples=SAMPLES,
        evals=1,
        seconds=1e6
    )
end

function benchmark_mnl_nhts_dataframe()
    data = read_gzipped_csv(get_test_file_path("nhts/hhpub.csv.gz"), DataFrame)

    data.hhveh_topcode = min.(data.HHVEHCNT, 4)

    @benchmarkable(
        multinomial_logit(
            @utility(begin
                0 ~ α0
                1 ~ α1 + β1homeown * (HOMEOWN == 2) + @dummy_code(β1, HH_RACE, [2, 3, 4, 5, 6, 97]) + β1_hhsize * HHSIZE
                2 ~ α2 + β2homeown * (HOMEOWN == 2) + @dummy_code(β2, HH_RACE, [2, 3, 4, 5, 6, 97]) + β2_hhsize * HHSIZE
                3 ~ α3 + β3homeown * (HOMEOWN == 2) + @dummy_code(β3, HH_RACE, [2, 3, 4, 5, 6, 97]) + β3_hhsize * HHSIZE
                4 ~ α4plus + β4plushomeown * (HOMEOWN == 2) + @dummy_code(β4plus, HH_RACE, [2, 3, 4, 5, 6, 97]) + β4plus_hhsize * HHSIZE

                α0 = 0, fixed
            end),
            :hhveh_topcode,
            $data
        ),
        samples=SAMPLES,
        evals=1,
        seconds=1e6
    )
end

function benchmark_mnl_nhts_macroexpand()
    @benchmarkable(@macroexpand(@utility(begin
                0 ~ α0
                1 ~ α1 + β1homeown * (HOMEOWN == 2) + @dummy_code(β1, HH_RACE, [2, 3, 4, 5, 6, 97]) + β1_hhsize * HHSIZE
                2 ~ α2 + β2homeown * (HOMEOWN == 2) + @dummy_code(β2, HH_RACE, [2, 3, 4, 5, 6, 97]) + β2_hhsize * HHSIZE
                3 ~ α3 + β3homeown * (HOMEOWN == 2) + @dummy_code(β3, HH_RACE, [2, 3, 4, 5, 6, 97]) + β3_hhsize * HHSIZE
                4 ~ α4plus + β4plushomeown * (HOMEOWN == 2) + @dummy_code(β4plus, HH_RACE, [2, 3, 4, 5, 6, 97]) + β4plus_hhsize * HHSIZE

                α0 = 0, fixed
            end)),
            samples=SAMPLES,
            seconds=1e6,
            evals=1)
end

function should_run(benchmark_name)
    run = length(ARGS) == 0 || benchmark_name in ARGS
    # INFO level debugging is disabled. If I were fancy I'd filter log messages by module, but... I'm not
    run || @warn "skipping benchmark $benchmark_name"
    return run
end


function main()
    "--verbose" in ARGS || Logging.disable_logging(Logging.Info)
    suite = BenchmarkGroup()

    should_run("MNL_Swissmetro_DTable") && (suite["MNL_Swissmetro_DTable"] = benchmark_mnl_dtable())
    should_run("MNL_Swissmetro_DataFrame") && (suite["MNL_Swissmetro_DataFrame"] = benchmark_mnl())
    should_run("MNL_Swissmetro_macroexpand") && (suite["MNL_Swissmetro_macroexpand"] = benchmark_mnl_macroexpand())
    should_run("MNL_NHTS_DTable") && (suite["MNL_NHTS_DTable"] = benchmark_mnl_nhts_dtable())
    should_run("MNL_NHTS_DataFrame") && (suite["MNL_NHTS_DataFrame"] = benchmark_mnl_nhts_dataframe())
    should_run("MNL_NHTS_macroexpand") && (suite["MNL_NHTS_macroexpand"] = benchmark_mnl_nhts_macroexpand())


    #tune!(suite)
    results = run(suite, verbose=true)

    show(IOContext(stdout, :compact => false), results)
end

# ugly that this has to be done at top level
addprocs()

@everywhere begin
    using Pkg
    Pkg.activate(Base.source_dir())
    using DiscreteChoiceModels
end

main()