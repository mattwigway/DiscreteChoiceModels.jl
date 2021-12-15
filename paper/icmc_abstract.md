---
bibliography: icmc_abstract.bib
csl: apa-6th-edition.csl
---

# DiscreteChoiceModels.jl: High-performance scalable discrete choice models in Julia

Julia is a relatively new high-level dynamic programming language for numerical computing, with performance approaching C [@bezanson_julia_2017]. This article introduces `DiscreteChoiceModels.jl`, a new open-source package for estimating discrete choice models in Julia.

`DiscreteChoiceModels.jl` is has an intuitive syntax for specifying models, allowing users to directly write out their utility functions. For instance, the code below specifies the Swissmetro example mode-choice distributed with Biogeme [@bierlaire_short_2020]:

```julia
multinomial_logit(
    @utility(begin
        1 ~ αtrain + βtravel_time * TRAIN_TT / 100 + βcost * (TRAIN_CO * (GA == 0)) / 100
        2 ~ αswissmetro + βtravel_time * SM_TT / 100 + βcost * SM_CO * (GA == 0) / 100
        3 ~ αcar + βtravel_time * CAR_TT / 100 + βcost * CAR_CO / 100

        αswissmetro = 0, fixed
    end),
    :CHOICE,
    data,
    availability=[
        1 => :avtr,
        2 => :avsm,
        3 => :avcar,
    ]
)
```

Within the utility function specification (`@utility`), the first three lines specify the utility functions for each of the three modes specified by the CHOICE variable: train, car, and the hypothetical Swissmetro. Any variable starting with α or β (easily entered in Julia as `\alpha` and `\beta`) is treated as a coefficient to be estimated, while other variables are assumed to be data columns. The final line specifies that the ASC for Swissmetro should have a starting value of 0, and be fixed rather than estimated. The remainder of the model specification indicates that the choice is indicated by the variable `CHOICE`, what data to use, and, optionally, what columns indicate availability for each alternative.

## Features

`DiscreteChoiceModels.jl` currently supports estimating multinomial logit models; support for nested and mixed logit models, as well as prediction, is forthcoming. All optimization methods in `Optim.jl` [@mogensen_optim_2018] are supported, including BFGS (the default), BHHH, Newton's method, and Gradient Descent. Derivatives for optimization and for computation of variance-covariance matrices are exactly calculated using automatic differentiation [@revels_forward_2016], providing both performance and accuracy improvements over finite-difference approximations. Data can be read using either `DataFrames.jl` (most common), or `Dagger`, which provides the ability to scale model estimation across multiple nodes in a compute cluster. Both backends allow scaling across cores within a single machine.

To help ensure algorithm correctness, `DiscreteChoiceModels.jl` has an automated test suite that compares estimation results against ground-truth results for the same models from other software. This test suite is run automatically on each change to the `DiscreteChoiceModels.jl` source code.

## Performance

Julia is designed for high-performance computing, so a major goal of `DiscreteChoiceModels.jl` is to estimate models more quickly than other modeling packages. To that end, two models were developed and benchmarked using three packages---`DiscreteChoiceModels.jl`, Biogeme [@bierlaire_short_2020], and Apollo [@hess_apollo_2019], using default settings for all three packages. The first model is the Swissmetro example from Biogeme, with 6,768 observations, 3 alternatives, and 4 free parameters. The second is a vehicle ownership model using the 2017 US National Household Travel Survey, with 129,696 observations, 5 alternatives, and 35 free parameters. All runtimes are the median of 10 runs, and executed serially on a lightly-loaded circa-2014 quad-core Intel i7 with 16GB of RAM, running Debian Linux 11.1. `DiscreteChoiceModels.jl` outperforms the other packages when used with a DataFrame; using `Dagger` is slower due to the overhead of using a distributed computing system for a small model on a single machine.

Model                   DiscreteChoiceModels.jl: DataFrame    DiscreteChoiceModels.jl: Dagger   Biogeme           Apollo  
-----------------       ----------------------------------    --------------------------------  -------------     ---------
Swissmetro              188ms                                 2047ms                            252ms              824ms
Vehicle ownership       35.1s                                 46.9s                             163.4s             227.2s
-----------------       ----------------------------------    --------------------------------  -------------     ---------

Table 1: Comparison of model runtimes from `DiscreteChoiceModels.jl` and other packages. Julia runtimes include time to interpret the model specification, but not time to compile the `DiscreteChoiceModels.jl` package.

## Scalability

For extremely large models, a single machine may not be powerful enough to estimate the model, either due to RAM or processing constraints. Using the `Dagger` backend and Julia's built-in distributed computing capabilities, it is possible to scale model estimation across multiple nodes in a compute cluster. This is expected to be especially valuable for computationally-intensive mixed logit models.

Others have implemented this by modifications to the optimization algorithm [@shi_distributed_2019;@gopal_distributed_2013]. `DiscreteChoiceModels.jl` takes a simpler approach. Data are divided into chunks for each node in the cluster. For a given set of parameters, the log-likelihood of each chunk is computed. These are transmitted back to the main node where they are summed to produce the overall log-likelihood. This approach was also used by @zwaenepoel_inference_2019 in a model of gene duplication in tree species.

## References