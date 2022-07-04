# The NHTS example model used for the "harder" benchmark

import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, bioDraws, exp, MonteCarlo, log
from benchmarkable import Benchmarkable
import os
import biogeme.draws as draws

class ApolloMixed(Benchmarkable):
    def setup(self):
        # Read the data
        df = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__), "../../../test/data/apollo_swissRouteChoiceData.csv"
            )
        )

        self.database = db.Database("swissRouteChoice", df)

        self.dv = self.database.variables

        def normal_halton7(sampleSize, numberOfDraws):
            unif = draws.getHaltonDraws(
                sampleSize, numberOfDraws, base=7, skip=10
            )
            return draws.getNormalWichuraDraws(
                sampleSize,
                numberOfDraws,
                uniformNumbers=unif,
                antithetic=False,
            )

        mydraws = {"NORMAL_HALTON7": (normal_halton7, "Normal halton draws, Base 7")}
        self.database.setRandomNumberGenerators(mydraws)

    def measurable(self):
        # for comparability with DCM.jl, do all model setup here, but create 
        # derived variables outside the timed loop
        # Parameters to be estimated
        B_tt_mean = Beta("tt_mean", -3, None, None, 0)
        B_tt_sd = Beta("tt_sd", -0.01, None, None, 0)
        B_tc_mean = Beta("tc_mean", -3, None, None, 0)
        B_tc_sd = Beta("tc_sd", -0.01, None, None, 0)
        B_hw_mean = Beta("hw_mean", -3, None, None, 0)
        B_hw_sd = Beta("hw_sd", -0.01, None, None, 0)
        B_ch_mean = Beta("ch_mean", -3, None, None, 0)
        B_ch_sd = Beta("ch_sd", -0.01, None, None, 0)
        
        B_tt = -exp(B_tt_mean + B_tt_sd * bioDraws("B_tt", "NORMAL_HALTON2"))
        B_tc = -exp(B_tc_mean + B_tc_sd * bioDraws("B_tc", "NORMAL_HALTON3"))
        B_hw = -exp(B_hw_mean + B_hw_sd * bioDraws("B_hw", "NORMAL_HALTON5"))
        B_ch = -exp(B_ch_mean + B_ch_sd * bioDraws("B_ch", "NORMAL_HALTON7"))

        # Definition of the utility functions
        V1 = B_tt * self.dv["tt1"] + B_tc * self.dv["tc1"] + B_hw * self.dv["hw1"] + B_ch * self.dv["ch1"]
        V2 = B_tt * self.dv["tt2"] + B_tc * self.dv["tc2"] + B_hw * self.dv["hw2"] + B_ch * self.dv["ch2"]


        # Associate utility functions with the numbering of alternatives
        V = {1: V1, 2: V2}
        
        av = {1: 1, 2: 1}

        # Definition of the model. This is the contribution of each
        # observation to the log likelihood function.
        prob = models.logit(V, av, self.dv["choice"])
        logprob = log(MonteCarlo(prob))

        # Create the Biogeme object
        self.biogeme = bio.BIOGEME(self.database, logprob, numberOfDraws=500)
        self.biogeme.modelName = "apollo_mixed"

        # actually do the estimation
        self.biogeme.estimate()

        return self.biogeme
