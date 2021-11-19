# Run benchmarks in Python
# In the past I've had bad experiences running more than one Biogeme model in a single Python session
# Thus, run each in its own process

from biogeme_swissmetro import BiogemeSwissmetro

if __name__ == "__main__":
    bsmn = 100
    bsm = BiogemeSwissmetro.benchmark(bsmn)
    print(f"Biogeme Swissmetro: median {bsm}secs ({bsmn} executions)")
