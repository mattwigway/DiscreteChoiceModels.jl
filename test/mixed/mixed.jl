@testitem "Apollo route choice" begin
    using CSV, DataFrames, Distributions, StatsBase

    in_confint(coef, coef2, se2) = coef > coef2 - 1.96 * se2 && coef < coef2 + 1.96 * se2

    # The route choice mixed logit example model from Apollo
    # http://www.apollochoicemodelling.com/files/examples/6%20Mixture%20models/MMNL_preference_space.r
    # http://www.apollochoicemodelling.com/files/examples/6%20Mixture%20models/output/MMNL_preference_space_output.txt
    data = CSV.read(joinpath(Base.source_dir(), "../data/apollo_swissRouteChoiceData.csv"), DataFrame)

    # specify the model, using the same starting values as in the Apollo example
    model = mixed_logit(
        @utility(begin
            1 ~ -βtt * tt1 + -βtc * tc1 + -βhw * hw1 + -βch * ch1
            2 ~ -βtt * tt2 + -βtc * tc2 + -βhw * hw2 + -βch * ch2

            # The example uses standard deviations of -0.01
            # (sign can be negative b/c of how the lognormal is constructed in Apollo).
            # exp(-4.6502) ≈ 0.01
            βtt = LogNormal(-3, exp(-4.6052)), level=>ID
            βtc = LogNormal(-3, exp(-4.6052)), level=>ID
            βhw = LogNormal(-3, exp(-4.6052)), level=>ID
            βch = LogNormal(-3, exp(-4.6052)), level=>ID
        end),
        :choice,
        data,
        verbose=:medium,
        draws=500
    )

    # ensure the coefficients are close
    coefs = Dict(zip(coefnames(model), coef(model)))
    ses = Dict(zip(coefnames(model), stderror(model)))
    # test that the coefficients we found are in the confidence intervals from Apollo
    # just checking confidence intervals b/c of simulation differences
    # You might initially assume that they should be exactly the same because Apollo and DCM.jl
    # both use 500 Halton draws and BFGS, but the critical difference is that Apollo works by constructing
    # standard normal draws and then multiplying/adding to adjust mean and sd, while we estimate mean/sd
    # directly. This means that any slight asymmetry to the draws can be reversed in Apollo, but not in
    # DCM.jl. We are also estimating the log of the standard error to prevent it from going negative.
    @test in_confint(coefs[:βtt_μ], -1.9843, 0.08812)
    @test in_confint(exp(coefs[:βtt_log_σ]), 0.4662, 0.07753)
    @test in_confint(coefs[:βtc_μ], -1.0246, 0.13792)
    @test in_confint(exp(coefs[:βtc_log_σ]), 0.9843, 0.09134)
    @test in_confint(coefs[:βhw_μ], -2.9339, 0.08465)
    @test in_confint(exp(coefs[:βhw_log_σ]), 0.8163, 0.11757)
    @test in_confint(coefs[:βch_μ], 0.6234, 0.07391)
    @test in_confint(exp(coefs[:βch_log_σ]), 0.8263, 0.12050)

    @test loglikelihood(model) > -1445 && loglikelihood(model) < -1440
    @test isapprox(model.init_ll, -2253.78, rtol=0.01)
    @test isapprox(nullloglikelihood(model), -2420.39, rtol=0.01)
end
