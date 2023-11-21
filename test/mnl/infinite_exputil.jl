# In the original, naive implementation of the multinomial logit likelihood, we just calculated exp(util) / sum(exp.(utils))
# but this can lead to overflow when util is large. This didn't seem to affect convergence, somehow, but is
# still undesirable. Now we use the softmax function from LogExpFunctions which handles overflow.

@testitem "Infinite exp(util)" begin
    ll = DiscreteChoiceModels.mnl_ll_row((ch=1.0,), [50000.0, 2.0], [(p, r, _) -> p[1], (p, r, _) -> p[2]], Val(:ch), nothing)
    @test isfinite(ll)
end