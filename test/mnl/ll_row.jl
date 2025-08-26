@testitem "MNL LL Row" begin
    import DiscreteChoiceModels: mnl_ll_row

    @test mnl_ll_row(
        (x=1.0, y=1.2, chosen=1),
        [1.0, 1.2, 1.4, 1.5, 1.3],
        (
            (params, row, _) -> params[1] * row.x + params[2] * row.y,
            (params, row, _) -> params[3] + params[4] * row.x + params[5] * row.y,
        ),
        Val(:chosen),
        Val(nothing)
    ) ≈
        # this is computing this the naive way, using exponentiation and then taking the log. The function does
        # something a bit smarter and does everything in log space, to avoid overflow
        log(exp(1.0 * 1.0 + 1.2 * 1.2) / (exp(1.0 * 1.0 + 1.2 * 1.2) + exp(1.4 + 1.5 * 1.0 + 1.3 * 1.2)))
end

@testitem "Extreme values in mnl_ll_row" begin
    import DiscreteChoiceModels: mnl_ll_row

    # test that we do not have overflow problems when the utilities are really large
    @test mnl_ll_row(
        (chosen=1,),
        Float64[],
        (
            # these numbers are way too big to exponentiate without overflow
            (_, _, _) -> 1_000_000,
            (_, _, _) -> 1_000_000
        ),
        Val(:chosen),
        Val(nothing)
    ) ≈ log(0.5)

    # test that we do not have overflow problems when the probabilities are really small
    for digits in 10:50:5000
        util, logprob1, logprob2 = setprecision(8192) do
            expected_probability = big"10" ^ BigFloat(-digits)
            (
                convert(Float64, log(expected_probability) - log(big"1" - expected_probability)),
                convert(Float64, log(expected_probability)),
                convert(Float64, log(big"1" - expected_probability))
            )
        end

        @test mnl_ll_row(
            (chosen=1,),
            Float64[],
            (
                (_, _, _) -> util,
                (_, _, _) -> 0.0
            ),
            Val(:chosen),
            Val(nothing)
        ) ≈ logprob1

        # We previously had a test for the components where the probability was close to 100% as well,
        # but that failed. In this case, there are two outcomes, one with utility -x (x large) and one with
        # utility 0. The log choice probability is computed directly in log space as util - ln ∑ exp(util).
        #
        # We do use the logsumexp operation to avoid overflow in logs, but I still get some roundoff errors
        # in the probabilities of very common options when there are very uncommon options. For instance, when
        # one option has probability 1e-50 and the other has probability 1-1e-50, I get the correct probability
        # estimate for the less common outcome, but the probability for the most common outcome rounds to 1. This
        # is because I'm computing the log-probability in log space, so the utility in the numerator of the likelihood
        # function is never exponentiated. But the numbers in the denominator do get exponentiated, even though they
        # are shifted from the logsumexp, so there is roundoff even though there isn't overflow. But this roundoff
        # will only affect large probabilities, which should not have much effect on the loglikelihood since they are
        # close to 1, so being off by 1e-50 is a tiny fraction, whereas being off by 1e-50 when the true probability is 2e-50
        # is a factor of 2 error in likelihood.

        @test mnl_ll_row(
            (chosen=2,),
            Float64[],
            (
                (_, _, _) -> util,
                (_, _, _) -> 0.0
            ),
            Val(:chosen),
            Val(nothing)
        ) ≈ logprob2 atol=1e-10 # atol 1e-10 in log space = factor of 100 parts per trillion
    end
end
