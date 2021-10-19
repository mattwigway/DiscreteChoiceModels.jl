module DiscreteChoiceModels

include("LogitModel.jl")
include("util.jl")
include("utilitymacro.jl")
include("mnl.jl")
include("compute_utility.jl")

export multinomial_logit, @utility

end