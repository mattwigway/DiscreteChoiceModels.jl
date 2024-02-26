import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, bioDraws, exp, MonteCarlo, log
from biogeme.nests import OneNestForNestedLogit, NestsForNestedLogit
from benchmarkable import Benchmarkable
import os

class MlogitNested(Benchmarkable):
    def setup(self):
        # Read the data
        df = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__),"../../../test/data/mlogit_hc.csv"
            )
        )

        df["idx_choice"] = df.depvar.replace({
            "gcc": 1,
            "ecc": 2,
            "erc": 3,
            "hpc": 4,
            "gc": 5,
            "ec": 6,
            "er": 7
        })

        df = df.rename(columns=lambda x: x.replace(".", "_"))

        self.database = db.Database("heating", df.drop(columns="depvar"))

        self.dv = self.database.variables

    def measurable(self):
        # for comparability with DCM.jl, do all model setup here, but create 
        # derived variables outside the timed loop
        # Parameters to be estimated
        ASC_cooling = Beta("ASC_cooling", 0, None, None, 0)
        B_heat_install_cost = Beta("B_heat_install_cost", 0, None, None, 0)
        B_heat_op_cost = Beta("B_heat_op_cost", 0, None, None, 0)
        B_cool_install_cost = Beta("B_cool_install_cost", 0, None, None, 0)
        B_cool_op_cost = Beta("B_cool_op_cost", 0, None, None, 0)
        B_inc_cooling = Beta("B_inc_cooling", 0, None, None, 0)
        B_inc_room = Beta("B_inc_room", 0, None, None, 0)

        MU_cool = Beta("MU_cool", 1, None, None, 0)
        MU_nocool = Beta("MU_nocool", 1, None, None, 0)

        # COOLING
        # Gas central heat with cooling
        V_gcc = ASC_cooling + B_heat_install_cost * self.dv["ich_gcc"] + B_heat_op_cost * self.dv["och_gcc"] + B_cool_install_cost * self.dv["icca"] + B_cool_op_cost * self.dv["occa"] + B_inc_cooling * self.dv["income"]
        # Electric central resistance heat with cooling
        V_ecc = ASC_cooling + B_heat_install_cost * self.dv["ich_ecc"] + B_heat_op_cost * self.dv["och_ecc"] + B_cool_install_cost * self.dv["icca"] + B_cool_op_cost * self.dv["occa"] + B_inc_cooling * self.dv["income"]
        # Electric room resistance heat with cooling
        V_erc = ASC_cooling + B_heat_install_cost * self.dv["ich_erc"] + B_heat_op_cost * self.dv["och_erc"] + B_cool_install_cost * self.dv["icca"] + B_cool_op_cost * self.dv["occa"] + B_inc_cooling * self.dv["income"] + B_inc_room * self.dv["income"]
        # Electric heat pump (implies cooling)
        V_hpc = ASC_cooling + B_heat_install_cost * self.dv["ich_hpc"] + B_heat_op_cost * self.dv["och_hpc"] + B_cool_install_cost * self.dv["icca"] + B_cool_op_cost * self.dv["occa"] + B_inc_cooling * self.dv["income"]

        # NON-COOLING
        # Gas central heat without cooling
        V_gc = B_heat_install_cost * self.dv["ich_gc"] + B_heat_op_cost * self.dv["och_gc"]
        # Electric central resistance heat without cooling
        V_ec = B_heat_install_cost * self.dv["ich_ec"] + B_heat_op_cost * self.dv["och_ec"]
        # Electric room resistance heat without cooling
        V_er = B_heat_install_cost * self.dv["ich_er"] + B_heat_op_cost * self.dv["och_er"] + B_inc_room * self.dv["income"]


        # Associate utility functions with the numbering of alternatives
        V = {1: V_gcc, 2: V_ecc, 3: V_erc, 4: V_hpc, 5: V_gc, 6: V_ec, 7: V_er}

        # Associate the availability conditions with the alternatives
        av = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}

        # Set up the nests
        cooling = OneNestForNestedLogit(nest_param=MU_cool, list_of_alternatives=[1, 2, 3, 4])
        nocooling = OneNestForNestedLogit(nest_param=MU_nocool, list_of_alternatives=[5, 6, 7])
        nests = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=(cooling, nocooling))

        # Definition of the model. This is the contribution of each
        # observation to the log likelihood function.
        logprob = models.lognested(V, av, nests, self.dv["idx_choice"])

        # Create the Biogeme object
        self.biogeme = bio.BIOGEME(self.database, logprob)
        self.biogeme.modelName ="01logit"

        # actually do the estimation
        return self.biogeme.estimate()

        # code to get non-robust standard errors for comparison
        # vcov = result.getVarCovar()
        # pd.Series(np.sqrt(np.diag(vcov).astype("float64")), vcov.columns.values)