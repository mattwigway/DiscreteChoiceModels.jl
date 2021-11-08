using Distributed, Logging

#=
This function initializes workers with the JuliaDB and JuliaDBMeta packages.
If you initialize workers another way, you do not need this function.

This may modify the environment on the workers. This is fine for workers that
are one-time-use, but may not be desirable in all cluster environments.
=#
function init_workers()
    @info "Initializing $(workers()) workers"
    @everywhere begin
        using Pkg
        Pkg.add("JuliaDB")
        Pkg.add("JuliaDBMeta")
        Pkg.precompile()
    end
end