@testset "@β macro" begin
    #=
    @test MacroTools.striplines(@macroexpand @β(["commonwealth"], ["massachusetts", "virginia"])) ==
        MacroTools.striplines(:(βcommonwealth_massachusetts * massachusetts + βcommonwealth_virginia * virginia))

    @test MacroTools.striplines(@macroexpand @β(["massachusetts", "virginia", "kentucky", "pennsylvania"], "commonwealth")) ==
        MacroTools.striplines(:(
            βcommonwealth_massachusetts * massachusetts +
            βcommonwealth_virginia * virginia +
            βcommonwealth_kentucky * kentucky +
            βcommonwealth_pennsylvania * pennsylvania
        ))

    @test MacroTools.striplines(@macroexpand @β(["carolina"], "north")) ==
        MacroTools.striplines(:(βnorth_carolina * carolina))
        
    @test MacroTools.striplines(@macroexpand @β([], "south")) == MacroTools.striplines(quote end)
    =#
end
