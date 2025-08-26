"""
    @foreachchoice [choice1, choice2, choice3....] α + βx1 * x1 + βx2 * x2

Repeat the specified utility function for each specified choice. All α and β values will be postfixed with choice.
"""
macro foreachchoice(choices, expr)
    output = []

    for choice in choices.args
        choiceutil = postwalk(copy(expr)) do x
            if x isa Symbol && iscoef(x)
                αorβ, coefname = match(r"^([βαθ])(.*)$", String(x))
                return Symbol("$(αorβ)$(choice)$(isempty(coefname) ? "" : "_")$(coefname)")
            else
                return x
            end
        end

        # esc has to go here, otherwise julia will replace ~ with DiscreteChoiceModels.~
        push!(output, esc(:($choice ~ $choiceutil)))
    end

    # https://discourse.julialang.org/t/how-to-create-a-function-from-a-vector-of-expressions/73457/2
    return Expr(:block, output...)
end