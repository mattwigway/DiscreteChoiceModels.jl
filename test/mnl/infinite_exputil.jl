# In the original, naive implementation of the multinomial logit likelihood, we just calculated exp(util) / sum(exp.(utils))
# but this can lead to overflow when util is large. This didn't seem to affect convergence, somehow, but is
# still undesirable. Now we use the logsumexp function from LogExpFunctions which avoids overflow.

@testitem "Infinite exp(util)" begin
    import LogExpFunctions: softmax

    ll = DiscreteChoiceModels.mnl_ll_row((ch=1,), [50000.0, 2.0], [(p, r, _) -> p[1], (p, r, _) -> p[2]], Val(:ch), nothing)
    @test isfinite(ll)
    @test exp(ll) ≈ softmax([50000.0, 2.0])[1]

    # also should not allocate (seems like @view allocates - investigate)
    # alloc = @allocated DiscreteChoiceModels.mnl_ll_row(row, params, ufuncs, chcol, nothing, nalts)
    # @test alloc == 0

    # test with availability constraints - there is now a third option, but it's not available,
    # so this should not change anything
    ll = DiscreteChoiceModels.mnl_ll_row((ch=1, av1=true, av2=true, av3=false), [50000.0, 2.0, 4.0],
        [(p, r, _) -> p[1], (p, r, _) -> p[2], (p, r, _) -> p[3]], Val(:ch), [:av1, :av2, :av3])
    @test isfinite(ll)
    @test exp(ll) ≈ softmax([50000.0, 2.0])[1]
end