# supertype for all logit model results

using StatsBase

abstract type LogitModel <: RegressionModel end

StatsBase.coefnames(r::LogitModel) = r.coefnames
StatsBase.coef(r::LogitModel) = r.coefs
StatsBase.islinear(r::LogitModel) = false
StatsBase.loglikelihood(r::LogitModel) = r.final_ll
StatsBase.nullloglikelihood(r::LogitModel) = r.const_ll
StatsBase.vcov(r::LogitModel) = r.vcov
StatsBase.stderror(r::LogitModel) = sqrt.(diag(StatsBase.vcov(r)))