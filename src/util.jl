#=
Convert a vector of pairs of {choice => availaility} to a matrix
if availability is nothing, return nothing
=#
function availability_to_matrix(availability::Union{Nothing, <:AbstractVector{<:Pair{<:Any, <:AbstractVector{Bool}}}}, alt_numbers::Dict{<:Any, Int64})::Union{BitMatrix, Nothing}
    if isnothing(availability)
        return nothing
    else
        avail_mat = BitMatrix(undef, length(availability[1].second), length(alt_numbers))
        for (name, avvec) in availability
            index = alt_numbers[name]
            @assert !isnothing(index)
            avail_mat[:, index] = avvec
        end

        return avail_mat
    end
end