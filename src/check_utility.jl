"Run various checks on the utility object to make sure it is sensible"
function check_utility(utility, data)
    check_utility_allocations(utility, data)
end

"Warn if the utility function allocates"
function check_utility_allocations(utility, data)
    # todo abstract for distributed
    row = first(Tables.namedtupleiterator(data))
    for (alt_name, index) in pairs(utility.alt_numbers)
        # run it once to get it compiles
        # TODO handle mixed logit
        ufunc = utility.utility_functions[index]
        ufunc(utility.starting_values, row, nothing)

        # TODO: for now, seems they always allocate 16 bytes
        if (bytes_alloced = @allocated ufunc(utility.starting_values, row, nothing)) > 16
            @warn """Utility function for alternative $alt_name allocated $bytes_alloced bytes.
            
If this is unexpected, and you have a complicated utility function, the problem may be that you have run out of registers.
Addition with floating-point numbers is not commutative, so for a utility function a + bx + cy, Julia will calculate a, bx
and cy individually, then add them together. Adding parentheses around parts of your utility function may solve the problem.
"""
        end
    end
end