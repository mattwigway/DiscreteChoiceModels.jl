import MacroTools
using MacroTools: postwalk, @capture

# symbols that should not be interpreted as variable names - must be a better way to do this
const RESERVED_SYMBOLS = Set([:+, :*, :/, :-, :^, :(==), :(!=), :<, :>, :(<=), :(>=), :≤, :≥])

#=
define a utility function like so:
    @utility begin
        outcome ~ :asc_1 + :b_car * car
        :asc_1 = 2
        :b_car = 3f
    end

Symbols are treated as coefficients. Starting values for coefficients are defined with equal signs. Fixed values should be postfixed with f.
Coefficients that are not defined will be treated as estimated with a starting value of 0
=#
macro utility(ex::Expr)
    # first, extract coefficient definitions
    # turn coefs into a vector, reference them by integer values
    coefnames = Vector{Symbol}()
    starting_values = Vector{Float64}()
    coef_indices = Dict{Symbol, Int64}()
    fixed_coefs = Dict{Symbol, Number}()
    postwalk(ex) do subex
        if @capture(subex, (:coef_ = (starting_val_, fixed)))
            !(starting_val isa Number) && error("Fixed starting value must be a number, for coef $coef")
            # fixed coefficient
            (haskey(coef_indices, coef) || haskey(fixed_coefs, coef)) && error("Coef $coef defined multiple times")
            fixed_coefs[coef] = starting_val
        elseif @capture(subex, :coef_ = starting_val_)
            !(starting_val isa Number) && error("Starting value must be a number, for coef $coef")
            # non-fixed coefficient
            (haskey(coef_indices, coef) || haskey(fixed_coefs, coef)) && error("Coef $coef defined multiple times")
            push!(coefnames, coef)
            push!(starting_values, convert(Float64, starting_val))
            coef_indices[coef] = length(coefnames)
        end

        return subex  # make sure we don't mangle the expression while we're iterating over it
    end

    # now find all utility functions, and number alternatives
    util_funcs = Vector{Expr}()
    alt_numbers = Dict{Any, Int64}()
    postwalk(ex) do subex
        if @capture(subex, lhs_ ~ rhs_)
            # turn the right hand side into a function of the params
            parsed_rhs = postwalk(rhs) do x
                if @capture(x, :coef_)
                    # convert :coef to coefs - either references into the params array for
                    # non-fixed coefs, or literal values for fixed coefs
                    if haskey(fixed_coefs, coef)
                        # interpolate in literal value
                        return :($(fixed_coefs[coef]))
                    elseif haskey(coef_indices, coef)
                        # interpolate in reference into params array
                        return :(params[$(coef_indices[coef])])
                    else
                        # coef not specifically defined, create it implicitly with starting value 0
                        push!(coefnames, coef)
                        push!(starting_values, zero(Float64))
                        coef_indices[coef] = length(coefnames)
                        return :(params[$(length(coefnames))])
                    end
                elseif (x isa Symbol) && !in(x, RESERVED_SYMBOLS)
                    # convert bare values to data values
                    return :(row.$x)
                else
                    return x
                end
            end

            # turn alternatives into numbers for processing speed
            push!(util_funcs, :((params, row) -> $parsed_rhs))
            alt_numbers[lhs] = length(util_funcs)
        end

        return subex
    end

    # TODO some kind of error checking that there aren't other expressions that people meant to be
    # utility function or coefficient definitions, but didn't parse correctly

    return quote
        (
            coefnames = $coefnames,
            starting_values = $starting_values,
            fixed_coefs = $fixed_coefs,
            utility_functions = [$(util_funcs...)],
            alt_numbers = $alt_numbers
        )
    end
end