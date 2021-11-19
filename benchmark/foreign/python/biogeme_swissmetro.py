# Biogeme's Swissmetro example: https://github.com/michelbierlaire/biogeme/blob/master/examples/swissmetro/01logit.ipynb
# this code is copied and lightly modified from the Biogeme example, which is under the Biogeme open-source license

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta
from benchmarkable import Benchmarkable
import os


class BiogemeSwissmetro(Benchmarkable):
    def setup(self):
        # Read the data
        df = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__), "../../../test/data/biogeme_swissmetro.dat"
            ),
            sep="\t",
        )
        self.database = db.Database("swissmetro", df)

        self.dv = dv = self.database.variables

        # Removing some observations
        exclude = (
            (dv["PURPOSE"] != 1) * (dv["PURPOSE"] != 3) + (dv["CHOICE"] == 0)
        ) > 0
        self.database.remove(exclude)

        # Definition of new variables
        self.SM_COST = dv["SM_CO"] * (dv["GA"] == 0)
        self.TRAIN_COST = dv["TRAIN_CO"] * (dv["GA"] == 0)
        self.CAR_AV_SP = dv["CAR_AV"] * (dv["SP"] != 0)
        self.TRAIN_AV_SP = dv["TRAIN_AV"] * (dv["SP"] != 0)
        self.TRAIN_TT_SCALED = dv["TRAIN_TT"] / 100
        self.TRAIN_COST_SCALED = self.TRAIN_COST / 100
        self.SM_TT_SCALED = dv["SM_TT"] / 100
        self.SM_COST_SCALED = self.SM_COST / 100
        self.CAR_TT_SCALED = dv["CAR_TT"] / 100
        self.CAR_CO_SCALED = dv["CAR_CO"] / 100

    def measurable(self):
        # for comparability with DCM.jl, do all model setup here, but create 
        # derived variables outside the timed loop
        # Parameters to be estimated
        ASC_CAR = Beta("ASC_CAR", 0, None, None, 0)
        ASC_TRAIN = Beta("ASC_TRAIN", 0, None, None, 0)
        ASC_SM = Beta("ASC_SM", 0, None, None, 1)
        B_TIME = Beta("B_TIME", 0, None, None, 0)
        B_COST = Beta("B_COST", 0, None, None, 0)

        # Definition of the utility functions
        V1 = ASC_TRAIN + B_TIME * self.TRAIN_TT_SCALED + B_COST * self.TRAIN_COST_SCALED
        V2 = ASC_SM + B_TIME * self.SM_TT_SCALED + B_COST * self.SM_COST_SCALED
        V3 = ASC_CAR + B_TIME * self.CAR_TT_SCALED + B_COST * self.CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V = {1: V1, 2: V2, 3: V3}

        # Associate the availability conditions with the alternatives
        av = {1: self.TRAIN_AV_SP, 2: self.dv["SM_AV"], 3: self.CAR_AV_SP}

        # Definition of the model. This is the contribution of each
        # observation to the log likelihood function.
        logprob = models.loglogit(V, av, self.dv["CHOICE"])

        # Create the Biogeme object
        self.biogeme = bio.BIOGEME(self.database, logprob)
        self.biogeme.modelName = "01logit"

        # actually do the estimation
        self.biogeme.estimate()
