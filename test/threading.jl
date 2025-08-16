@testitem "Threading" begin
    import DataFrames: DataFrame
    import DiscreteChoiceModels: rowwise_loglik

    # 101 is prime, won't divide evenly
    df = DataFrame(:x=>1:101)

    for chunks in 1:200
        # we use a custom loglik_for_row that ensures that every row is included exactly once in log-likelihood calculations
        # tests are not _actually_ multithreaded so this is safe
        rows_encountered = Set{Int64}()

        duplicate = false

        rowwise_loglik(df, Int64[]; chunks=chunks) do row, _
            # make sure we don't encounter more than once
            # putting tests in closures doesn't seem to work
            duplicate = duplicate || row.x ∈ rows_encountered
            push!(rows_encountered, row.x)

            # gotta return something for it to sum
            return 0
        end

        @test !duplicate
        @test rows_encountered == Set{Int64}(1:101)
    end
end