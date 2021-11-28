#=
Functions to compute likelihood based on DataFrames or JuliaDB tables
=#

# find a prefix that is not used by any columns in the table by appending
# numbers to prefix
function find_unused_prefix(table, prefix::String)
    names = string.(Tables.columnnames(table))
    return find_unused_prefix(names, prefix)
end

# function find_unused_prefix(table::JuliaDB.AbstractIndexedTable, prefix::String)
#     names = string.(colnames(table))
#     return find_unused_prefix(names, prefix)
# end

function find_unused_prefix(names::Union{Vector{String}, NTuple{N, String}}, prefix::String) where N
    name = prefix
    idx = 0
    while any(startswith.(names, [name]))
        name = @sprintf "%s_%d" prefix idx
        idx += 1
    end
    return Symbol(name)
end

# common code to process availability into a vector of which column
# indicates availability for each numbered choice
function index_availability(availability, alt_numbers)
    if isnothing(availability)
        avail_cols = nothing
    else
        avail_cols = convert(Vector{Union{Missing, keytype(alt_numbers)}}, fill(missing, length(alt_numbers)))
        for (alt, avail_col) in availability
            idx = alt_numbers[alt]
            ismissing(avail_cols[idx]) || error("availability for $alt multiply defined, or duplicate alt indices")
            avail_cols[idx] = avail_col
        end

        any(ismissing.(avail_cols)) && error("incomplete availability matrix")
    end

    return avail_cols
end

# TODO change signature to generic Tables.jl table
function prepare_data(table::DataFrame, chosen, alt_numbers, availability)
    # copying input data, not great
    output_table = DataFrame(table)
    
    # find an unused name
    choice_col = find_unused_prefix(output_table, "enumerated_choice")

    output_table[!, choice_col] = get.([alt_numbers], output_table[:, chosen], [-1])
    any(output_table[!, choice_col] .== -1) && error("not all alternatives appear in utility functions")

    avail_cols = index_availability(availability, alt_numbers)
    return output_table, choice_col, avail_cols
end

function prepare_data(table::DTable, chosen, alt_numbers, availability)
    # find an unused name
    choice_col = find_unused_prefix(table, "enumerated_choice")

    table = map(r -> merge(NamedTuple{(choice_col,), Tuple{Int64}}(alt_numbers[r[chosen]]), pairs(r)), table)

    avail_cols = index_availability(availability, alt_numbers)
    return table, choice_col, avail_cols
end

#=
Version of prepare_data to work with JuliaDB IndexedTables
Does exactly the same as the above function, using multiple dispatch
to seamlessly handle different data sources. Should work out of core
automatically on a Julia cluster.
function prepare_data(table::JuliaDB.AbstractIndexedTable, chosen, alt_numbers, availability)
    # find an unused name
    choice_col = find_unused_prefix(table, "enumerated_choice")

    # TODO ensure that alt_numbers, chosen are available on remote workers
    # smaller concern - make sure they don't leak memory by remaining in memory
    # (they're small, so not a huge concern).
    # TODO does this involve a lot of copying?
    # TODO probably don't need an Int64 here
    # Constructing NamedTuple manually since the field names are determined at runtime
    output_table = @transform table NamedTuple{(choice_col,), Tuple{Int64}}((alt_numbers[cols(chosen)],))
    reduce(min, output_table; select=(choice_col => Int64)) < 0 && error("not all alternatives appear in utility functions")

    avail_cols = index_availability(availability, alt_numbers)
    return output_table, choice_col, avail_cols
end
=#

# All tables.jl sources converted to DataFrame in prepare_data above
function rowwise_loglik(loglik_for_row::Function, table::DataFrame, params::Vector{<:Any}, args...)
    # make the vector the same as the element type of params so ForwardDiff works
    thread_ll = zeros(eltype(params), Threads.nthreads())
    for row in Tables.namedtupleiterator(table)
        thread_ll[Threads.threadid()] += loglik_for_row(row, params, args...)
    end

    return sum(thread_ll)
end

#=
as above, but for a (possibly-distributed) IndexedTable
=#
function rowwise_loglik(loglik_for_row, table::DTable, params::Vector{T}, args...) where T <: Number
    # fingers crossed forwarddiff can handle distributed functions, I think it should
    # most function calls use first defn, first call on each worker uses second
    # TODO ensure enclosure here is type-stable
    # reducer(x::T, y::NamedTuple)::T = x + loglik_for_row(y, params, args...)::T
    # reducer(x::NamedTuple, y::NamedTuple)::T = loglik_for_row(x, params, args...)::T + loglik_for_row(y, params, args...)::T
    # reducer(x::T, y::T)::T = (x + y)::T

    return fetch(reduce(+, map(r -> (ll=loglik_for_row(r, params, args...),), table))).ll
end