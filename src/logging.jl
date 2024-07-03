"""
    write_header(io, utility)

Write the header for an estimation log file.
"""
function write_log_header(io::IO, utility)
    cols = ["iteration", "time", "objective", "gradient_norm", String.(utility.coefnames)...]
    # build the line
    row = join(("\"$(escape_string(c))\"" for c in cols), ',')
    println(io, row)
end

"""
    iteration_callback(io, verbose, optimstate)

Print an update on the estimation progress, and log to file if io != nothing.
"""
function iteration_callback(io::Union{IO, Nothing}, verbose, optimstate)
    time = get(Dates.now, optimstate.metadata, "time")

    if verbose
        @info "Iteration $(optimstate.iteration) (time $time): log-likelihood $(-optimstate.value), gradient norm $(optimstate.g_norm)"
    end

    if !isnothing(io)
        x = if haskey(optimstate.metadata, "x")
            optimstate.metadata["x"]
        elseif haskey(optimstate.metadata, "centroid")
            # Nelder-Mead
            optimstate.metadata["centroid"]
        else
            @warn "Optimization method does not provide coefficients at each iteration but logging requested"
            []
        end

        println(io, join([optimstate.iteration, time, -optimstate.value, optimstate.g_norm, x...], ','))
        flush(io)
    end

    # continue iteration
    return false
end
