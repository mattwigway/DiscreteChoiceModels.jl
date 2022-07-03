import MacroTools
using MacroTools: postwalk, @capture

# symbols that should not be interpreted as variable names - must be a better way to do this
const RESERVED_SYMBOLS = Set([:+, :*, :/, :-, :^, :(==), :(!=), :<, :>, :(<=), :(>=), :≤, :≥])

iscoef(coef::Symbol) = startswith(String(coef), r"[βα]")
getcoef(coef::QuoteNode) = getcoef(coef.value)
getcoef(coef::Symbol) = iscoef(coef) ? coef : error("unable to parse coef $coef")

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
    # inner macros are not parsed first, so expand all of them (e.g. @β)
    ex = macroexpand(Main, ex)

    # first, extract coefficient definitions
    # turn coefs into a vector, reference them by integer values
    coefnames = Vector{Symbol}()
    starting_values = Vector{Float64}()
    coef_indices = Dict{Symbol, Int64}()
    fixed_coefs = Dict{Symbol, Number}()

    # coefficients that require simulation
    mixed_coefs = Vector{Expr}()
    mixed_coefnames = Vector{Symbol}()
    mixed_coefindices = Dict{Symbol, Int64}()
    mixed_levels = Vector{Union{Symbol, Nothing}}() # this contains the level of aggregation

    postwalk(ex) do subex
        if @capture(subex, coefnode_ = starting_val_)
            (coefnode isa Symbol) && (startswith(String(coefnode), r"[αβ]")) || error("Coefficient name expected, but $coefnode found. Maybe you used = instead of ~ when defining a utility function?")
            coef = getcoef(coefnode)
            (haskey(coef_indices, coef) || haskey(fixed_coefs, coef) || haskey(mixed_coefindices, coef)) &&
                error("Coef $coef defined multiple times")

            if @capture(starting_val, (fixed_value_, fixed))
                # fixed coefficient
                !(fixed_value isa Number) && error("Fixed starting value must be a number, for coef $coef")
                fixed_coefs[coef] = fixed_value

            elseif starting_val isa Number
                # non-fixed coefficient
                (haskey(coef_indices, coef) || haskey(fixed_coefs, coef)) && error("Coef $coef defined multiple times")
                push!(coefnames, coef)
                # TODO hacky manual conversion, use automatic promotion
                push!(starting_values, convert(Float64, starting_val))
                coef_indices[coef] = length(coefnames)

            elseif @capture(starting_val, (distr_(dparams__) | distr_(dparams__), level=>lvlvar_))
                # mixed coefficient (i.e. distribution)
                n_params = length(dparams)
                
                # special case param names for common distributions
                paramlabels = if distr == :(Normal) || distr == :(LogNormal)
                    ["μ", "σ"][1:n_params] # allow mean or scale to be omitted
                elseif distr == Uniform
                    n_params == 0 ? String[] : ["min", "max"]
                else
                    ["param_$i" for i in 1:n_params]
                end
                
                # arguments to the distribution function that will be in the returned function
                distr_args = Vector{Expr}()

                # now, figure out which ones were fixed
                for (param, label) in zip(dparams, paramlabels)
                    if @capture(param, (fixed_val_, fixed))
                        (fixed_val isa Number) || error("Distribution fixed parameter $(coef)_$label must be number, was $fixed_val")
                        name = Symbol("$(coef)_$label")
                        (haskey(coef_indices, name) || haskey(fixed_coefs, name) || haskey(mixed_coefindices, name)) &&
                            error("Coef $name defined multiple times")
                        fixed_coefs[name] = fixed_val
                        push!(distr_args, :($fixed_val))
                    elseif @capture(param, exp(log_val_))
                        (log_val isa Number) || error("Distribution parameter $(coef)_$label must be number, was $param")
                        name = Symbol("$(coef)_log($label)")
                        (haskey(coef_indices, name) || haskey(fixed_coefs, name) || haskey(mixed_coefindices, name)) &&
                            error("Coef $name defined multiple times")
                        push!(coefnames, name)
                        coef_indices[name] = length(coefnames)
                        push!(starting_values, log_val)
                        push!(distr_args, :(exp(params[$(length(coefnames))])))
                    else
                        (param isa Number) || error("Distribution parameter $(coef)_$label must be number, was $param")
                        name = Symbol("$(coef)_$label")
                        (haskey(coef_indices, name) || haskey(fixed_coefs, name) || haskey(mixed_coefindices, name)) &&
                            error("Coef $name defined multiple times")
                        push!(coefnames, name)
                        coef_indices[name] = length(coefnames)
                        push!(starting_values, param)
                        push!(distr_args, :(params[$(length(coefnames))]))
                    end
                end

                push!(mixed_coefs, :(params -> $(distr)($(distr_args...))))
                push!(mixed_coefnames, coef)
                push!(mixed_levels, lvlvar)  # will be nothing for intraindividual draws
                mixed_coefindices[coef] = length(mixed_coefs)
            else
                error("Starting value must be a number or distribution, for coef $coef")
            end
        end

        return subex  # make sure we don't mangle the expression while we're iterating over it
    end

    # now find all utility functions, and number alternatives
    util_funcs = Vector{Expr}()
    alt_numbers = Dict{Any, Int64}()
    columns = Set{Any}()
    postwalk(ex) do subex
        if @capture(subex, lhs_ ~ rhs_)
            # turn the right hand side into a function of the params
            parsed_rhs = postwalk(rhs) do x
                if x isa Symbol
                    if iscoef(x)
                        # convert :coef to coefs - either references into the params array for
                        # non-fixed coefs, or literal values for fixed coefs
                        if haskey(fixed_coefs, x)
                            # interpolate in literal value
                            return :($(fixed_coefs[x]))
                        elseif haskey(coef_indices, x)
                            # interpolate in reference into params array
                            return :(params[$(coef_indices[x])])
                        elseif haskey(mixed_coefindices, x)
                            return :(mixed_coefs[$(mixed_coefindices[x])])
                        else
                            # coef not specifically defined, create it implicitly with starting value 0
                            push!(coefnames, x)
                            push!(starting_values, zero(Float64))
                            coef_indices[x] = length(coefnames)
                            return :(params[$(length(coefnames))])
                        end
                    elseif Base.isoperator(x)
                        # TODO would be useful to also allow things like trig functions here. Maybe we need
                        # something like I(...) like they have in R.
                        return x
                    else
                        # convert bare values to data values
                        push!(columns, x)
                        return :(row.$x)
                    end
                else
                    return x
                end
            end

            # turn alternatives into numbers for processing speed
            push!(util_funcs, :(function (params::Vector{T}, row, mixed_coefs::Union{AbstractVector{T}, Nothing}=nothing) where T <: Number
                # ensure the return value is always a T, even when the function is e.g. a constant 0
                convert(T, @inbounds($parsed_rhs))
            end))
            alt_numbers[lhs] = length(util_funcs)
        end

        return subex
    end

    # TODO some kind of error checking that there aren't other expressions that people meant to be
    # utility function or coefficient definitions, but didn't parse correctly

    # figure out groupcol
    groupcol = if isempty(mixed_levels) || all(isnothing.(mixed_levels))
        nothing
    else
        groupcols = unique(filter(x -> !isnothing(x), mixed_levels))
        length(groupcols) == 1 || error("Multiple levels of aggregation not supported!")
        first(groupcols)
    end

    return quote
        (
            coefnames = $coefnames,
            starting_values = $starting_values,
            fixed_coefs = $fixed_coefs,
            utility_functions = [$(util_funcs...)],
            alt_numbers = $alt_numbers,
            columnnames=$columns,
            mixed_coefs=[$(mixed_coefs...)],
            mixed_coefnames=$mixed_coefnames,
            mixed_levels=$mixed_levels,
            # cannot interpolate symbol directly or it will try to be looked up
            # https://stackoverflow.com/questions/48272986
            groupcol=$(groupcol isa Symbol ? Meta.quot(groupcol) : groupcol)
        )
    end
end

"""
Create betas for all of the names in the vector
"""
macro β(outcomenode, variablenode, includealpha=false)
    outcomes = eval(outcomenode)
    vars = eval(variablenode)
    
    res = Vector{Expr}()
    for outcome in outcomes
        full, short = if outcome isa Pair
            outcome[1], outcome[2]
        else
            outcome, outcome
        end

        expressions = map(vars) do varn
            β = (Symbol("β$(short)_$varn"))
            x = Symbol(varn)
            :($β * $x)
        end

        if includealpha
            expressions = convert(Vector{Union{Expr, Symbol}}, expressions)
            pushfirst!(expressions, Symbol("α$short"))
        end

        rhs = if length(expressions) == 0
            quote end
        elseif length(expressions) == 1
            expressions[1]
        else
            Expr(:call, :+, expressions...)
        end

        push!(res, :($full ~ $rhs))
    end

    return esc(quote $(res...) end)
end

macro dummy_code(prefix, column, valuenode)
    values = eval(valuenode)
    expressions = map(values) do val
        β = (Symbol("$(prefix)_$(column)_$(val)"))
        :($β * ($column == $val))
    end

    if length(expressions) == 0
        quote end
    elseif length(expressions) == 1
        esc(expressions[1])
    else
        esc(Expr(:call, :+, expressions...))
    end
end