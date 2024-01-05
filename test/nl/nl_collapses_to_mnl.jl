# When the inclusive value parameter is 1 (which is its starting value),
# the nested logit collapses to the multinomial logit.
@testitem "Nested logit collapses to multinomial logit" begin
    using DataFrames
    df = DataFrame(
        :choice => ["SOV", "Transit", "SOV", "Carpool", "Transit"],
        :income => [75, 45, 75, 55, 65]
    )

    println(@utility(begin
        "SOV" ~ 0
        "Transit" ~ βinc_tr * income
        "Carpool" ~ βinc_cp * income
    end))

    mnl = multinomial_logit(
        @utility(begin
            "SOV" ~ 0
            "Transit" ~ βinc_tr * income
            "Carpool" ~ βinc_cp * income
        end),
        :choice,
        df
    )

    nl = nested_logit(
        @utility(begin
            "SOV" ~ 0
            "Transit" ~ βinc_tr * income
            "Carpool" ~ βinc_cp * income
            "Drive" ~ 0

            "Drive" => ["SOV", "Carpool"]
        end),
        :choice,
        df
    )

    # at starting values, they should collapse to the same model
    @test mnl.init_ll ≈ nl.init_ll
end