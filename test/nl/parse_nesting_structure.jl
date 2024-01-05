@testitem "Parse simple nesting structure" begin
    # test a simple (two level) nesting structure
    import DiscreteChoiceModels: TOP_LEVEL_NEST, parse_nesting_structure!

    # Test that parsing nesting structure works correctly
    alt_numbers = Dict(
        "PT" => 1,
        "Train" => 2,
        "Bus" => 3,
        "Drive" => 4,
        "SOV" => 5,
        "Carpool" => 6,
        "Walk" => 7
    )

    nests = fill(TOP_LEVEL_NEST, length(alt_numbers))

    parse_nesting_structure!(:("PT" => ["Train", "Bus"]), nests, alt_numbers)
    parse_nesting_structure!(:("Drive" => ["SOV", "Carpool"]), nests, alt_numbers)

    # PT and Drive are not in a nest, train and bus are in PT, SOV and Carpool are in Drive, Walk is not in a nest/degenerate nest
    @test nests == [TOP_LEVEL_NEST, 1, 1, TOP_LEVEL_NEST, 4, 4, TOP_LEVEL_NEST]
end

@testitem "Parse nested nesting structure" begin
    # here we have a nesting structure that looks like this:
    # PT
    #  - Train
    #  - Bus
    #    - Rapid
    #    - Local
    #      - With seat
    #      - Standing room only
    # Drive
    #  - SOV
    #  - Carpool
    # Walk
    # 
    # I would not recommend actually using a nesting structure this complicated - it might
    # result in convergence issues - but it is a good test of the parser
    import DiscreteChoiceModels: TOP_LEVEL_NEST, parse_nesting_structure!

    alt_numbers = Dict(
        "PT" => 1,
        "Train" => 2,
        "Bus" => 3,
        "Rapid" => 4,
        "Local" => 5,
        "With seat" => 6,
        "Standing room only" => 7,
        "Drive" => 8,
        "SOV" => 9,
        "Carpool" => 10,
        "Walk" => 11
    )

    nests = fill(TOP_LEVEL_NEST, length(alt_numbers))
    parse_nesting_structure!(:("PT" => [
        "Train",
        "Bus" => [
            "Local" => ["With seat", "Standing room only"],
            "Rapid"
        ]]),
        nests, alt_numbers)

    parse_nesting_structure!(:("Drive" => ["SOV", "Carpool"]), nests, alt_numbers)

    @test nests == [TOP_LEVEL_NEST, 1, 1, 3, 3, 5, 5, TOP_LEVEL_NEST, 8, 8, TOP_LEVEL_NEST]
end