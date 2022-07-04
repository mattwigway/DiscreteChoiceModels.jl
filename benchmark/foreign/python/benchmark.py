# Run benchmarks in Python
# In the past I've had bad experiences running more than one Biogeme model in a single Python session
# Thus, run each in its own process

from biogeme_swissmetro import BiogemeSwissmetro
from biogeme_nhts import NHTS
from apollo_mixed import ApolloMixed

if __name__ == "__main__":
    bsmn = 5
    bsm = BiogemeSwissmetro.benchmark(bsmn)
    print(f"Biogeme Swissmetro: median {bsm}secs ({bsmn} executions)")

    bnhts = NHTS.benchmark(bsmn)
    print(f"Biogeme NHTS: median {bnhts}secs ({bsmn} executions)")

    bap = ApolloMixed.benchmark(bsmn)
    print(f"Biogeme Apollo Mixed: median {bap}secs ({bsmn} executions)")