# This tests that the scaled expected maximum utility (Γ in Koppelman and Bhat, 1/μ in Ben-Akiva and Lerman) is calculated correctly
@testitem "Expected maximum utility" begin
    import DiscreteChoiceModels: get_scaled_emu, TOP_LEVEL_NEST

    nests = [5, 5, 6, 6, TOP_LEVEL_NEST, TOP_LEVEL_NEST]
    # no top-level utility
    ufuncs = map(x -> ((_, _, _) -> x), [-1, -2, 3, -40, 0, 0])
    iv_param_indices = [TOP_LEVEL_NEST, TOP_LEVEL_NEST, TOP_LEVEL_NEST, TOP_LEVEL_NEST, 1, 2]
    params = [0.3, 0.8]

    # a leaf node should have 0 emu
    @test get_scaled_emu(nothing, params, iv_param_indices, ufuncs, nests, nothing, 1) ≈ 0.0

    # first nest
    @test get_scaled_emu(nothing, params, iv_param_indices, ufuncs, nests, nothing, 5) ≈ 0.3 * log(sum(exp.([-1, -2] ./ 0.3)))

    # second nest
    @test get_scaled_emu(nothing, params, iv_param_indices, ufuncs, nests, nothing, 6) ≈ 0.8 * log(sum(exp.([3, -40] ./ 0.8)))
end
