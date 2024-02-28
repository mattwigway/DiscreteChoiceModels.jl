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
        nothing
    else
        avail_cols = convert(Vector{Union{Missing, keytype(alt_numbers)}}, fill(missing, length(alt_numbers)))
        for (alt, avail_col) in availability
            idx = alt_numbers[alt]
            ismissing(avail_cols[idx]) || error("availability for $alt multiply defined, or duplicate alt indices")
            avail_cols[idx] = avail_col
        end

        any(ismissing.(avail_cols)) && error("incomplete availability matrix")
        tuple(avail_cols...)
    end
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
    !isnothing(avail_cols) && check_availability(output_table, avail_cols, choice_col)

    return output_table, choice_col, avail_cols
end

function prepare_data(table::DTable, chosen, alt_numbers, availability)
    # find an unused name
    choice_col = find_unused_prefix(table, "enumerated_choice")

    table = map(table) do r
        merge(NamedTuple{(choice_col,), Tuple{Int64}}(alt_numbers[r[chosen]]), pairs(r))
    end

    avail_cols = index_availability(availability, alt_numbers)
    !isnothing(avail_cols) && check_availability(table, avail_cols, choice_col)

    return table, choice_col, avail_cols
end

function check_availability(table, avail_cols, choice_col)
    for (i, row) in enumerate(Tables.rows(table))
        reduce(|, map(c -> row[c], avail_cols)) || @error "At row $i, no alternatives are available"
        row[avail_cols[row[choice_col]]] || @error "At row $i, chosen alternative is not available"
    end
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

"""
Compute log-likelihood rowwise for a DataFrame, using multiple threads if available. This works by splitting
the table into one chunk per thread, and running the likelihood calculation on each chunk. It does this
rather than using an off-the-shelf parallel mapreduce (e.g. ThreadsX or FLoops) to avoid allocations, as
SplittablesBase.halve(NamedTupleIterator) causes a lot of allocations (I'm not quite sure how, but it seems to
copy the full dataset). This is tested and speeds computation on large models by roughly an order of magnitude.
"""
function rowwise_loglik(loglik_for_row, table::DataFrame, params::Vector{T}, args...)::T where T <: Number
    chunksize_per_thread = nrow(table) ÷ Threads.nthreads() + 1

    start = 1
    thread_ll = @sync map(1:Threads.nthreads()) do _
        subdf = Tables.namedtupleiterator(@view table[start:min(start + chunksize_per_thread - 1, nrow(table)),:])
        start += chunksize_per_thread
        Threads.@spawn mapreduce(r -> loglik_for_row(r, params, args...), +, $subdf, init=zero(T))
    end

    @assert start > nrow(table)

    sum(fetch.(thread_ll))
end

#=
as above, but for a (possibly-distributed) IndexedTable
=#
function rowwise_loglik(loglik_for_row, table::DTable, params::Vector{T}, args...)::T where T <: Number
    # fingers crossed forwarddiff can handle distributed functions, I think it should
    # most function calls use first defn, first call on each worker uses second
    # TODO ensure enclosure here is type-stable
    # reducer(x::T, y::NamedTuple)::T = x + loglik_for_row(y, params, args...)::T
    # reducer(x::NamedTuple, y::NamedTuple)::T = loglik_for_row(x, params, args...)::T + loglik_for_row(y, params, args...)::T
    # reducer(x::T, y::T)::T = (x + y)::T

    ll_parts = Vector{Dagger.EagerThunk}()
    sizehint!(ll_parts, length(table.chunks))
    for chunk in table.chunks
        push!(ll_parts, Dagger.spawn(rowwise_loglik, loglik_for_row, chunk, params, args...))
    end

    fetch(Dagger.spawn(+, ll_parts...))
end

# Loglikelihood computed by groups, e.g. in panel mixed logit model
function groupwise_loglik(loglik_for_group, table, params::Vector{T}, args...) where T
    ll_parts = zeros(T, Threads.nthreads())
    @sync begin
        row_number = 1
        for (groupnumber, group) in enumerate(table)
            Threads.@spawn ll_parts[Threads.threadid()] += loglik_for_group($group, $row_number, $groupnumber, params, args...)
            row_number += length(group)
        end
    end

    sum(ll_parts)
end