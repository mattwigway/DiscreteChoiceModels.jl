# supertype for all logit model results

using StatsBase

abstract type LogitModel <: RegressionModel end

StatsBase.coefnames(r::LogitModel) = r.coefnames
StatsBase.coef(r::LogitModel) = r.coefs
StatsBase.islinear(r::LogitModel) = false
StatsBase.loglikelihood(r::LogitModel) = r.final_ll
StatsBase.stderror(r::LogitModel) = r.ses