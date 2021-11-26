#=

Benchmark simple models

=#

using DiscreteChoiceModels, CSV, DataFrames, BenchmarkTools, Logging, JuliaDB, JuliaDBMeta, Optim, Distributed, CodecZlib

const SAMPLES = 10

get_test_file_path(file) = joinpath(Base.source_dir(), "../test/data/", file)

function benchmark_mnl()
    data = CSV.read(get_test_file_path("biogeme_swissmetro.dat"), DataFrame, delim='\t')
    data = data[in.(data.PURPOSE, [Set([1, 3])]) .& (data.CHOICE .!= 0), :]

    data.avtr = (data.TRAIN_AV .== 1) .& (data.SP .!= 0)
    data.avsm = data.SM_AV .== 1
    data.avcar = (data.CAR_AV .== 1) .& (data.SP .!= 0)

    # TODO ensure compilation time for the macro is included!!!! for a fair comparison.
    @benchmarkable multinomial_logit(
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
<<<<<<< HEAD
    ) samples=100

    return suite
=======
    ) samples=SAMPLES
>>>>>>> 10faba5 (benchmark improvements)
end

function benchmark_mnl_juliadb()
    data = loadtable(get_test_file_path("biogeme_swissmetro.dat"), delim='\t', distributed=true)

    # filter to wanted cases
    data = filter(data) do row
        ((row.PURPOSE == 1) | (row.PURPOSE == 3)) && row.CHOICE != 0
    end

    data = @transform data (
        avtr=(:TRAIN_AV == 1) && (:SP != 0),
        avsm=:SM_AV == 1,
        avcar=(:CAR_AV == 1) && (:SP != 0)
    )

    @benchmarkable multinomial_logit(
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
            if T <: JuliaDB.AbstractIndexedTable
                loadtable(path, kwargs...)
            elseif T <: DataFrame
                CSV.read(path, DataFrame, kwargs...)
            else
                error("Don't know how to read table into type $T")
            end
        end
    end
end

function benchmark_mnl_nhts_juliadb()
    data = read_gzipped_csv(get_test_file_path("nhts/hhpub.csv.gz"), JuliaDB.IndexedTable)

    data = filter(data) do row
        row.HHSTATE in Set(["CA", "OR", "WA", "NV", "ID", "AZ", "UT", "CO", "NM", "MT", "TX", "OK", "NE", "WY", "ND", "SD"])
    end

    # topcode hhvehcnt
    data = @transform data (hhveh_topcode = :HHVEHCNT ≥ 4 ? "4plus" : string(:HHVEHCNT),)
    
    @benchmarkable multinomial_logit(
        @utility(begin
            "0" ~ α0
            "1" ~ α1 + β1homeown * (HOMEOWN == 2) + @dummy_code(β1, HH_RACE, [2, 3, 4, 5, 6, 97]) + β1_hhsize * HHSIZE
            "2" ~ α2 + β2homeown * (HOMEOWN == 2) + @dummy_code(β2, HH_RACE, [2, 3, 4, 5, 6, 97]) + β2_hhsize * HHSIZE
            "3" ~ α3 + β3homeown * (HOMEOWN == 2) + @dummy_code(β3, HH_RACE, [2, 3, 4, 5, 6, 97]) + β3_hhsize * HHSIZE
            "4plus" ~ α4plus + β4plushomeown * (HOMEOWN == 2) + @dummy_code(β4plus, HH_RACE, [2, 3, 4, 5, 6, 97]) + β4plus_hhsize * HHSIZE

            α0 = 0, fixed
        end),
        :hhveh_topcode,
        $data;
        verbose=:medium
    ) samples=SAMPLES
end

function benchmark_mnl_nhts_dataframe()
    data = read_gzipped_csv(get_test_file_path("nhts/hhpub.csv.gz"), DataFrame)

    data = data[in.(data.HHSTATE, Ref(Set(["CA", "OR", "WA", "NV", "ID", "AZ", "UT", "CO", "NM", "MT", "TX", "OK", "NE", "WY", "ND", "SD"]))), :]

    data.hhveh_topcode = string.(min.(data.HHVEHCNT, 4))
    data[data.hhveh_topcode .== "4", :hhveh_topcode] .= "4plus"

    @benchmarkable multinomial_logit(
        @utility(begin
            "0" ~ α0
            "1" ~ α1 + β1homeown * (HOMEOWN == 2) + @dummy_code(β1, HH_RACE, [2, 3, 4, 5, 6, 97]) + β1_hhsize * HHSIZE
            "2" ~ α2 + β2homeown * (HOMEOWN == 2) + @dummy_code(β2, HH_RACE, [2, 3, 4, 5, 6, 97]) + β2_hhsize * HHSIZE
            "3" ~ α3 + β3homeown * (HOMEOWN == 2) + @dummy_code(β3, HH_RACE, [2, 3, 4, 5, 6, 97]) + β3_hhsize * HHSIZE
            "4plus" ~ α4plus + β4plushomeown * (HOMEOWN == 2) + @dummy_code(β4plus, HH_RACE, [2, 3, 4, 5, 6, 97]) + β4plus_hhsize * HHSIZE

            α0 = 0, fixed
        end),
        :hhveh_topcode,
        $data;
        verbose=:medium
    ) samples=SAMPLES
end

function should_run(benchmark_name)
    run = length(ARGS) == 0 || benchmark_name in ARGS
    # INFO level debugging is disabled. If I were fancy I'd filter log messages by module, but... I'm not
    run && @warn "running benchmark $benchmark_name"
    run || @error "skipping benchmark $benchmark_name"
    return run
end


function main()
    "--verbose" in ARGS || Logging.disable_logging(Logging.Info)
    suite = BenchmarkGroup()

    should_run("MNL_Swissmetro_JuliaDB") && (suite["MNL_Swissmetro_JuliaDB"] = benchmark_mnl_juliadb())
    should_run("MNL_Swissmetro_DataFrame") && (suite["MNL_Swissmetro_DataFrame"] = benchmark_mnl())
    should_run("MNL_NHTS_JuliaDB") && (suite["MNL_NHTS_JuliaDB"] = benchmark_mnl_nhts_juliadb())
    should_run("MNL_NHTS_DataFrame") && (suite["MNL_NHTS_DataFrame"] = benchmark_mnl_nhts_dataframe())

    tune!(suite)
    results = run(suite)
    # use median runtime as not skewed by initial compilation
    println(median(results))
end

# ugly that this has to be done at top level
addprocs()

@everywhere begin
    using Pkg
    Pkg.activate(Base.source_dir())
    using DiscreteChoiceModels
end

main()