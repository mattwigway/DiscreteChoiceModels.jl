# This runs the heating/cooling example from the mlogit R package, available at
# https://cran.r-project.org/web/packages/mlogit/vignettes/e2nlogit.html. Note that
# the results presented there use the un.nest.el option, which constrains the IV parameters
# to be the same across all nests. DCM.jl does not support this, so we use the variation of
# the model in the exercises where they do not constrain the IV parameters. The full set of
# coefficients for this model is not represented on that page, but are presented below:

# Call:
# mlogit(formula = depvar ~ ich + och + icca + occa + inc.room + 
#     inc.cooling + int.cooling | 0, data = HC, nests = list(cooling = c("gcc", 
#     "ecc", "erc", "hpc"), other = c("gc", "ec", "er")), un.nest.el = FALSE)

# Frequencies of alternatives:choice
#    ec   ecc    er   erc    gc   gcc   hpc 
# 0.004 0.016 0.032 0.004 0.096 0.744 0.104 

# bfgs method
# 4 iterations, 0h:0m:0s 
# g'(-H)^-1g =  1.18 
# last step couldn't find higher value 

# Coefficients :
#              Estimate Std. Error z-value  Pr(>|z|)    
# ich         -0.562283   0.146145 -3.8474 0.0001194 ***
# och         -0.895493   0.271861 -3.2939 0.0009880 ***
# icca        -0.267062   0.150310 -1.7767 0.0756103 .  
# occa        -1.338514   1.264215 -1.0588 0.2897042    
# inc.room    -0.381441   0.096658 -3.9463 7.937e-05 ***
# inc.cooling  0.259932   0.062085  4.1867 2.830e-05 ***
# int.cooling -4.821927   5.528796 -0.8721 0.3831277    
# iv:cooling   0.611529   0.188736  3.2401 0.0011947 ** 
# iv:other     0.378394   0.133617  2.8319 0.0046270 ** 
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# Log-Likelihood: -178.04

# These are not exactly the values that DCM.jl finds. I suspect this is because we are not using sequential estimation, or
# possibly because we are using autodiff. We actually find values that produce a slightly better log-likelihood. I've
# put these coefficients back into mlogit in R and it confirms they produce a better log-likelihood:

# mlogit(formula = depvar ~ ich + och + icca + occa + inc.room + 
#     inc.cooling + int.cooling | 0, data = HC, start = c(-0.55429, 
#     -0.86667, -0.22538, -1.1053, -0.37786, 0.25193, -6.0644, 
#     0.60098, 0.44599), nests = list(cooling = c("gcc", "ecc", 
#     "erc", "hpc"), other = c("gc", "ec", "er")), un.nest.el = FALSE)

# Frequencies of alternatives:choice
#    ec   ecc    er   erc    gc   gcc   hpc 
# 0.004 0.016 0.032 0.004 0.096 0.744 0.104 

# bfgs method
# 1 iterations, 0h:0m:0s 
# g'(-H)^-1g = 2.48E-07 
# gradient close to zero 

# Coefficients :
#              Estimate Std. Error z-value  Pr(>|z|)    
# ich         -0.554296   0.148967 -3.7209 0.0001985 ***
# och         -0.866681   0.267274 -3.2427 0.0011842 ** 
# icca        -0.225386   0.148099 -1.5219 0.1280437    
# occa        -1.105285   1.232791 -0.8966 0.3699477    
# inc.room    -0.377861   0.101257 -3.7317 0.0001902 ***
# inc.cooling  0.251932   0.061632  4.0877 4.357e-05 ***
# int.cooling -6.064312   5.558927 -1.0909 0.2753106    
# iv:cooling   0.600981   0.190557  3.1538 0.0016115 ** 
# iv:other     0.445949   0.274948  1.6219 0.1048159    
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# Log-Likelihood: -177.81

@testitem "mlogit heating/cooling" begin
    import CSV
    import DataFrames: DataFrame, rename!
    import StatsBase: coef, coefnames, stderror, loglikelihood

    data = CSV.File(joinpath(dirname(Base.source_path()), "../data/mlogit_hc.csv")) |> DataFrame

    # make column names Julia symbol compatible
    rename!(x -> replace(x, "." => "_"), data)

    model = nested_logit(
        @utility(begin
            # COOLING
            # Gas central heat with cooling
            "gcc" ~ αcooling + βheat_install_cost * ich_gcc + βheat_op_cost * och_gcc + βcool_install_cost * icca + βcool_op_cost * occa + βinc_cooling * income
            # Electric central resistance heat with cooling
            "ecc" ~ αcooling + βheat_install_cost * ich_ecc + βheat_op_cost * och_ecc + βcool_install_cost * icca + βcool_op_cost * occa + βinc_cooling * income
            # Electric room resistance heat with cooling
            "erc" ~ αcooling + βheat_install_cost * ich_erc + βheat_op_cost * och_erc + βcool_install_cost * icca + βcool_op_cost * occa + βinc_cooling * income + βinc_room * income
            # Electric heat pump (implies cooling)
            "hpc" ~ αcooling + βheat_install_cost * ich_hpc + βheat_op_cost * och_hpc + βcool_install_cost * icca + βcool_op_cost * occa + βinc_cooling * income

            # NON-COOLING
            # Gas central heat without cooling
            "gc" ~ βheat_install_cost * ich_gc + βheat_op_cost * och_gc
            # Electric central resistance heat without cooling
            "ec" ~ βheat_install_cost * ich_ec + βheat_op_cost * och_ec
            # Electric room resistance heat without cooling
            "er" ~ βheat_install_cost * ich_er + βheat_op_cost * och_er + βinc_room * income

            # Nest-level values
            "cooling" ~ 0 # TODO move cooling specific stuff here?
            "noncooling" ~ 0

            # Nesting structure
            "cooling" => ["gcc", "ecc", "erc", "hpc"]
            "noncooling" => ["gc", "ec", "er"]            
        end),
        :depvar,
        data
    )

    @test round(loglikelihood(model), digits=2) ≈ -177.81


    coefs = Dict(zip(coefnames(model), round.(coef(model), digits=3)))
    ses = Dict(zip(coefnames(model), round.(stderror(model), digits=3)))

    println(coefs)

    # These SEs are not the SEs reported by mlogit, but rather those reported by Biogeme. The mlogit ones
    # don't match biogeme or DCM.jl, but DCM.jl and Biogeme match. This may be because it appears
    # mlogit is using an estimated Hessian from BHHH, not an actual Hessian, and when I ran with these
    # starting values the algorithm didn't have many iterations to refine the Hessian.
    @test coefs[:βheat_install_cost] ≈ -0.554
    @test   ses[:βheat_install_cost] ≈ 0.144
    @test coefs[:βheat_op_cost] ≈ -0.867
    @test   ses[:βheat_op_cost] ≈ 0.240
    @test coefs[:βcool_install_cost] ≈ -0.225
    @test   ses[:βcool_install_cost] ≈ 0.111
    @test coefs[:βcool_op_cost] ≈ -1.105
    @test   ses[:βcool_op_cost] ≈ 1.040
    @test coefs[:βinc_room] ≈ -0.378
    @test   ses[:βinc_room] ≈ 0.100
    @test coefs[:βinc_cooling] ≈ 0.252
    @test   ses[:βinc_cooling] ≈ 0.052
    @test coefs[:αcooling] ≈ -6.064
    @test   ses[:αcooling] ≈ 4.843 # biogeme reports 4.842, close enough
    @test coefs[:θcooling] ≈ 0.601
    # Biogeme estimates mu, not theta, so I'm not sure what SEs should be here
    #@test   ses[:θcooling] ≈ 0.191
    @test coefs[:θnoncooling] ≈ 0.446
    #@test   ses[:θnoncooling] ≈ 0.275
    
    # @test coefs[:βheat_install_cost] ≈ -0.562283
    # @test   ses[:βheat_install_cost] ≈ 0.146145
    # @test coefs[:βheat_op_cost] ≈ -0.895493
    # @test   ses[:βheat_op_cost] ≈ 0.271861
    # @test coefs[:βcool_install_cost] ≈ -0.267062
    # @test   ses[:βcool_install_cost] ≈ 0.150310
    # @test coefs[:βcool_op_cost] ≈ -1.338514
    # @test   ses[:βcool_op_cost] ≈ 1.264215
    # @test coefs[:βinc_room] ≈ -0.381441
    # @test   ses[:βinc_room] ≈ 0.096658
    # @test coefs[:βinc_cooling] ≈ 0.259932
    # @test   ses[:βinc_cooling] ≈ 0.062085
    # @test coefs[:αcooling] ≈ -4.821927
    # @test   ses[:αcooling] ≈ 5.528796
    # @test coefs[:θcooling] ≈ 0.611529
    # @test   ses[:θcooling] ≈ 0.188736
    # @test coefs[:θnoncooling] ≈ 0.378394
    # @test   ses[:θnoncooling] ≈ 0.133617
end