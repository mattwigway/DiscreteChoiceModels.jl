# Run the NHTS example in R

library(apollo)
library(tictoc)

apollo_initialise()

apollo_control = list(
  modelName = "nhts",
  modelDescr = "NHTS example",
  indivID = "HOUSEID",
  outputDirectory = tempdir()
)

database = read.csv(gzfile("../../../test/data/nhts/hhpub.csv.gz"))

database$homeown_dum = database$HOMEOWN == 1
database$choice = vapply(database$HHVEHCNT, function (v) { return(min(v, 4))}, 4)

tic()  # start timing - defining betas etc

apollo_beta = c(
  ASC_0v = 0,
  
  ASC_1v = 0,
  B_homeown_1v = 0,
  B_hhrace_2_1v = 0,
  B_hhrace_3_1v = 0,
  B_hhrace_4_1v = 0,
  B_hhrace_5_1v = 0,
  B_hhrace_6_1v = 0,
  B_hhrace_97_1v = 0,
  B_hhsize_1v = 0,
  
  ASC_2v = 0,
  B_homeown_2v = 0,
  B_hhrace_2_2v = 0,
  B_hhrace_3_2v = 0,
  B_hhrace_4_2v = 0,
  B_hhrace_5_2v = 0,
  B_hhrace_6_2v = 0,
  B_hhrace_97_2v = 0,
  B_hhsize_2v = 0,
  
  ASC_3v = 0,
  B_homeown_3v = 0,
  B_hhrace_2_3v = 0,
  B_hhrace_3_3v = 0,
  B_hhrace_4_3v = 0,
  B_hhrace_5_3v = 0,
  B_hhrace_6_3v = 0,
  B_hhrace_97_3v = 0,
  B_hhsize_3v = 0,
  
  ASC_4v = 0,
  B_homeown_4v = 0,
  B_hhrace_2_4v = 0,
  B_hhrace_3_4v = 0,
  B_hhrace_4_4v = 0,
  B_hhrace_5_4v = 0,
  B_hhrace_6_4v = 0,
  B_hhrace_97_4v = 0,
  B_hhsize_4v = 0
)

apollo_fixed = c("ASC_0v")

apollo_inputs = apollo_validateInputs()

apollo_probabilities = function (apollo_beta, apollo_inputs, functionality="estimate") {
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))
  
  # probabilities
  P = list()
  
  # utilities
  V = list(
    V0 = ASC_0v,
    V1 = ASC_1v + B_homeown_1v * homeown_dum +
        B_hhrace_2_1v * (HH_RACE == 2) +
        B_hhrace_3_1v * (HH_RACE == 3) +
        B_hhrace_4_1v * (HH_RACE == 4) +
        B_hhrace_5_1v * (HH_RACE == 5) +
        B_hhrace_6_1v * (HH_RACE == 6) +
        B_hhrace_97_1v * (HH_RACE == 97) +
        B_hhsize_1v * HHSIZE,
    V2 = ASC_2v + B_homeown_2v * homeown_dum +
        B_hhrace_2_2v * (HH_RACE == 2) +
        B_hhrace_3_2v * (HH_RACE == 3) +
        B_hhrace_4_2v * (HH_RACE == 4) +
        B_hhrace_5_2v * (HH_RACE == 5) +
        B_hhrace_6_2v * (HH_RACE == 6) +
        B_hhrace_97_2v * (HH_RACE == 97) +
        B_hhsize_2v * HHSIZE,
    V3 = ASC_3v + B_homeown_3v * homeown_dum +
        B_hhrace_2_3v * (HH_RACE == 2) +
        B_hhrace_3_3v * (HH_RACE == 3) +
        B_hhrace_4_3v * (HH_RACE == 4) +
        B_hhrace_5_3v * (HH_RACE == 5) +
        B_hhrace_6_3v * (HH_RACE == 6) +
        B_hhrace_97_3v * (HH_RACE == 97) +
        B_hhsize_3v * HHSIZE,
    V4 = ASC_4v + B_homeown_4v * homeown_dum +
        B_hhrace_2_4v * (HH_RACE == 2) +
        B_hhrace_3_4v * (HH_RACE == 3) +
        B_hhrace_4_4v * (HH_RACE == 4) +
        B_hhrace_5_4v * (HH_RACE == 5) +
        B_hhrace_6_4v * (HH_RACE == 6) +
        B_hhrace_97_4v * (HH_RACE == 97) +
        B_hhsize_4v * HHSIZE
  )
  
  mnl_settings = list(
    alternatives=c(V0=0, V1=1, V2=2, V3=3, V4=4),
    avail = list(V0=1, V1=1, V2=1, V3=1, V4=1),
    choiceVar = choice,
    utilities = V
  )
  
  P[["model"]] = apollo_mnl(mnl_settings, functionality)
  
  P = apollo_prepareProb(P, apollo_inputs, functionality)
  
  return(P)
}

model = apollo_estimate(apollo_beta, apollo_fixed, apollo_probabilities, apollo_inputs)

timer = toc()

print(paste("Model estimation time:", (timer$toc - timer$tic)[[1]], "seconds"))
