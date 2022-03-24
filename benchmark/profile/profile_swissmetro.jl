using Distributed; addprocs()
@everywhere begin
    using Pkg
    Pkg.activate(Base.source_dir())
end

using DiscreteChoiceModels, Dagger, Tables, CSV, Infiltrator

function main()
    data = DTable(Tables.rowtable(CSV.File("test/data/biogeme_swissmetro.dat", delim='\t')), 2000)

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

    @time multinomial_logit(
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
end

main()