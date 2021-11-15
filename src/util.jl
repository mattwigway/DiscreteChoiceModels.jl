#=
Convert a vector of pairs of {choice => availaility} to a matrix
if availability is nothing, return nothing
=#
function availability_to_matrix(availability::Union{Nothing, <:AbstractVector{<:Pair{<:Any, <:AbstractVector{Bool}}}}, alt_numbers::Dict{<:Any, Int64})::Union{BitMatrix, Nothing}
    if isnothing(availability)
        return nothing
    else
        avail_mat = BitMatrix(undef, length(availability[1].second), length(alt_numbers))
        for (name, avvec) in availability
            index = alt_numbers[name]
            @assert !isnothing(index)
            avail_mat[:, index] = avvec
        end

        return avail_mat
    end
end

#=
Find unique values of a column
=#
function find_unique_values(table::JuliaDB.AbstractIndexedTable, column)
    reducer(x::NamedTuple, y::NamedTuple) = Set{typeof(x[column])}([x[column], y[column]])
    # treat sets as immutable. not sure if this matters but could imagine data races.
    reducer(x::Set, y::NamedTuple) = y[column] in x ? x : Set{typeof(y[column])}([x..., y[column]])
    reducer(x::Set, y::Set) = union(x, y)

    return reduce(reducer, table)
end

#=
Get column type for a column in an IndexedTable
=#
function coltype(table::JuliaDB.AbstractIndexedTable, column)
    ttyp = typeof(table)
    return ttyp.parameters[1].parameters[2].parameters[findfirst(p -> p == column, ttyp.parameters[1].parameters[1])]
end

#=
Evaluate perfect prediction problems, returns true if perfect prediction found
=#
function check_perfect_prediction(table::JuliaDB.AbstractIndexedTable, choice, predictors=collect(filter(c -> c != choice, colnames(table))))
    # first, find all unique values of choice
    unique_choice = find_unique_values(table, choice)

    problems_found = false

    # now, reduce to find min/max of each predictor for each choice
    # TODO this could be more efficient, in theory could call reduce for all cols at once, but I get a
    # JuliaDB error about namedtuples with duplicate names - I think due to using the same selector for
    # a bunch of OnlineStats
    for column in predictors
        ct = coltype(table, column)
        for (i, chc) in enumerate(unique_choice)
            minmax = groupreduce(Extrema(ct), table, choice => x -> x == chc; select=column) |>
                collect |>
                DataFrame

            minmax_true = minmax[minmax[:,choice], :Extrema][1]
            minmax_false = minmax[.!minmax[:,choice], :Extrema][1]

            if nobs(minmax_true) == 0
                @warn "Choice $chc not selected"
                problems_found = true
            elseif nobs(minmax_false) == 0
                @warn "Choice $chc always selected"
                problems_found = true
            elseif minimum(minmax_true) > maximum(minmax_false)
                @warn "Column $column > $(maximum(minmax_false)) perfectly predicts $choice == $chc"
                problems_found = true
            elseif maximum(minmax_true) < minimum(minmax_false)
                @warn "Column $column < $(minimum(minmax_false)) perfectly predicts $choice == $chc"
                problems_found = true
            end
        end

        vrnc = var(reduce(Variance(ct == Bool ? Float64 : ct), table, select=column))
        if vrnc < 1e-5
            @warn "Column $column has near-zero variance"
            problems_found = true
        end
    end

    return problems_found
end