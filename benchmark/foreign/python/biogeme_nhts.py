# The NHTS example model used for the "harder" benchmark

import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta
from benchmarkable import Benchmarkable
import os
import gzip


def gzip_to_pandas(filename, **kwargs):
    with gzip.open(filename, "rb") as inf:
        return pd.read_csv(inf, **kwargs)

class NHTS(Benchmarkable):
    def setup(self):
        # Read the data
        df = gzip_to_pandas(
            os.path.join(
                os.path.dirname(__file__), "../../../test/data/nhts/hhpub.csv.gz"
            )
        )

        df["homeown_dum"] = (df.HOMEOWN == 1).astype("float64")
        df["choice"] = np.minimum(df.HHVEHCNT, 4)

        df = df[[c for c, d in zip(df.dtypes.index, df.dtypes) if pd.api.types.is_numeric_dtype(d)]].copy()

        self.database = db.Database("nhts", df)

        self.dv = self.database.variables

    def measurable(self):
        # for comparability with DCM.jl, do all model setup here, but create 
        # derived variables outside the timed loop
        # Parameters to be estimated
        ASC_0v = Beta("ASC_0v", 0, None, None, 1)
        
        ASC_1v = Beta("ASC_1v", 0, None, None, 0)
        B_homeown_1v = Beta("B_homeown_1v", 0, None, None, 0)
        B_hhrace_2_1v = Beta("B_hhrace_2_1v", 0, None, None, 0)
        B_hhrace_3_1v = Beta("B_hhrace_3_1v", 0, None, None, 0)
        B_hhrace_4_1v = Beta("B_hhrace_4_1v", 0, None, None, 0)
        B_hhrace_5_1v = Beta("B_hhrace_5_1v", 0, None, None, 0)
        B_hhrace_6_1v = Beta("B_hhrace_6_1v", 0, None, None, 0)
        B_hhrace_97_1v = Beta("B_hhrace_97_1v", 0, None, None, 0)
        B_hhsize_1v = Beta("B_hhsize_1v", 0, None, None, 0)

        ASC_2v = Beta("ASC_2v", 0, None, None, 0)
        B_homeown_2v = Beta("B_homeown_2v", 0, None, None, 0)
        B_hhrace_2_2v = Beta("B_hhrace_2_2v", 0, None, None, 0)
        B_hhrace_3_2v = Beta("B_hhrace_3_2v", 0, None, None, 0)
        B_hhrace_4_2v = Beta("B_hhrace_4_2v", 0, None, None, 0)
        B_hhrace_5_2v = Beta("B_hhrace_5_2v", 0, None, None, 0)
        B_hhrace_6_2v = Beta("B_hhrace_6_2v", 0, None, None, 0)
        B_hhrace_97_2v = Beta("B_hhrace_97_2v", 0, None, None, 0)
        B_hhsize_2v = Beta("B_hhsize_2v", 0, None, None, 0)

        ASC_3v = Beta("ASC_3v", 0, None, None, 0)
        B_homeown_3v = Beta("B_homeown_3v", 0, None, None, 0)
        B_hhrace_2_3v = Beta("B_hhrace_2_3v", 0, None, None, 0)
        B_hhrace_3_3v = Beta("B_hhrace_3_3v", 0, None, None, 0)
        B_hhrace_4_3v = Beta("B_hhrace_4_3v", 0, None, None, 0)
        B_hhrace_5_3v = Beta("B_hhrace_5_3v", 0, None, None, 0)
        B_hhrace_6_3v = Beta("B_hhrace_6_3v", 0, None, None, 0)
        B_hhrace_97_3v = Beta("B_hhrace_97_3v", 0, None, None, 0)
        B_hhsize_3v = Beta("B_hhsize_3v", 0, None, None, 0)

        ASC_4v = Beta("ASC_4v", 0, None, None, 0)
        B_homeown_4v = Beta("B_homeown_4v", 0, None, None, 0)
        B_hhrace_2_4v = Beta("B_hhrace_2_4v", 0, None, None, 0)
        B_hhrace_3_4v = Beta("B_hhrace_3_4v", 0, None, None, 0)
        B_hhrace_4_4v = Beta("B_hhrace_4_4v", 0, None, None, 0)
        B_hhrace_5_4v = Beta("B_hhrace_5_4v", 0, None, None, 0)
        B_hhrace_6_4v = Beta("B_hhrace_6_4v", 0, None, None, 0)
        B_hhrace_97_4v = Beta("B_hhrace_97_4v", 0, None, None, 0)
        B_hhsize_4v = Beta("B_hhsize_4v", 0, None, None, 0)

        # Definition of the utility functions
        V0 = ASC_0v
        V1 = (
            ASC_1v + B_homeown_1v * self.dv["homeown_dum"] +
                B_hhrace_2_1v * (self.dv["HH_RACE"] == 2) +
                B_hhrace_3_1v * (self.dv["HH_RACE"] == 3) +
                B_hhrace_4_1v * (self.dv["HH_RACE"] == 4) +
                B_hhrace_5_1v * (self.dv["HH_RACE"] == 5) +
                B_hhrace_6_1v * (self.dv["HH_RACE"] == 6) +
                B_hhrace_97_1v * (self.dv["HH_RACE"] == 97) +
                B_hhsize_1v * self.dv["HHSIZE"]
        )
        V2 = (
            ASC_2v + B_homeown_2v * self.dv["homeown_dum"] +
                B_hhrace_2_2v * (self.dv["HH_RACE"] == 2) +
                B_hhrace_3_2v * (self.dv["HH_RACE"] == 3) +
                B_hhrace_4_2v * (self.dv["HH_RACE"] == 4) +
                B_hhrace_5_2v * (self.dv["HH_RACE"] == 5) +
                B_hhrace_6_2v * (self.dv["HH_RACE"] == 6) +
                B_hhrace_97_2v * (self.dv["HH_RACE"] == 97) +
                B_hhsize_2v * self.dv["HHSIZE"]
        )
        V3 = (
            ASC_3v + B_homeown_3v * self.dv["homeown_dum"] +
                B_hhrace_2_3v * (self.dv["HH_RACE"] == 2) +
                B_hhrace_3_3v * (self.dv["HH_RACE"] == 3) +
                B_hhrace_4_3v * (self.dv["HH_RACE"] == 4) +
                B_hhrace_5_3v * (self.dv["HH_RACE"] == 5) +
                B_hhrace_6_3v * (self.dv["HH_RACE"] == 6) +
                B_hhrace_97_3v * (self.dv["HH_RACE"] == 97) +
                B_hhsize_3v * self.dv["HHSIZE"]
        )
        V4 = (
            ASC_4v + B_homeown_4v * self.dv["homeown_dum"] +
                B_hhrace_2_4v * (self.dv["HH_RACE"] == 2) +
                B_hhrace_3_4v * (self.dv["HH_RACE"] == 3) +
                B_hhrace_4_4v * (self.dv["HH_RACE"] == 4) +
                B_hhrace_5_4v * (self.dv["HH_RACE"] == 5) +
                B_hhrace_6_4v * (self.dv["HH_RACE"] == 6) +
                B_hhrace_97_4v * (self.dv["HH_RACE"] == 97) +
                B_hhsize_4v * self.dv["HHSIZE"]
        )

        # Associate utility functions with the numbering of alternatives
        V = {0: V0, 1: V1, 2: V2, 3: V3, 4: V4}
        
        av = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}

        # Definition of the model. This is the contribution of each
        # observation to the log likelihood function.
        logprob = models.loglogit(V, av, self.dv["choice"])

        # Create the Biogeme object
        self.biogeme = bio.BIOGEME(self.database, logprob)
        self.biogeme.modelName = "nhts"

        # actually do the estimation
        self.biogeme.estimate()

        return self.biogeme
