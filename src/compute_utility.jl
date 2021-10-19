#=
Functions to compute likelihood based on DataFrames or JuliaDB tables
=#

using Tables
using JuliaDB
using Printf

# find a prefix that is not used by any columns in the table by appending
# numbers to prefix
# TODO should work on all tables, not just dataframes
function find_unused_prefix(table::DataFrame, prefix::String)
    names = string.(Tables.columnnames(table))
    name = prefix
    idx = 0
    while any(startswith.(names, [name]))
        name = @sprintf "%s_%d" prefix idx
        idx += 1
    end
    return Symbol(name)
end

# TODO change signature to generic Tables.jl table
function prepare_data(table::DataFrame, chosen, alt_numbers, availability)
    # copying input data, not great
    output_table = DataFrame(table)
    
    # find an unused name
    choice_col = find_unused_prefix(output_table, "enumerated_choice")

    output_table[!, choice_col] = get.([alt_numbers], output_table[:, chosen], [-1])
    any(output_table[!, choice_col] .== -1) && error("not all alternatives appear in utility functions")

    # add availability columns
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

    return output_table, choice_col, avail_cols
end

# All tables.jl sources converted to DataFrame in prepare_data above
function rowwise_loglik(loglik_for_row::Function, table::DataFrame, params::Vector{<:Any})
    # make the vector the same as the element type of params so ForwardDiff works
    thread_ll = zeros(eltype(params), Threads.nthreads())
    Threads.@threads for row in Tables.rows(table)
        thread_ll[Threads.threadid()] += loglik_for_row(row, params)
    end

    return sum(thread_ll)
end