# Run the Biogeme Swissmetro example in R

library(apollo)
library(tictoc)

apollo_initialise()

apollo_control = list(
  modelName = "biogeme_swissmetro",
  modelDescr = "Biogeme Swissmetro example",
  indivID = "ID",
  outputDirectory = tempdir()
)

# can't find a way to do source-relative paths in R
database = read.delim("../../../test/data/biogeme_swissmetro.dat")
database = subset(database, (PURPOSE == 1 | PURPOSE == 3) & CHOICE != 0)

database = within(database, {
  SM_COST = SM_CO * (GA == 0)
  TRAIN_COST = TRAIN_CO * (GA == 0)
  CAR_AV_SP = CAR_AV * (SP != 0)
  TRAIN_AV_SP = TRAIN_AV* (SP != 0)
  TRAIN_TT_SCALED = TRAIN_TT / 100
  TRAIN_COST_SCALED = TRAIN_COST / 100
  SM_TT_SCALED = SM_TT / 100
  SM_COST_SCALED = SM_COST / 100
  CAR_TT_SCALED = CAR_TT / 100
  CAR_CO_SCALED = CAR_CO / 100
})

# data loading and prep done, time the model setup and run
tic()

apollo_beta = c(
  asc_car = 0,
  asc_train = 0, 
  asc_sm = 0,
  b_time = 0,
  b_cost = 0
)

apollo_fixed = c("asc_sm")

apollo_inputs = apollo_validateInputs()

apollo_probabilities = function (apollo_beta, apollo_inputs, functionality="estimate") {
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))
  
  # probabilities
  P = list()
  
  # utilities
  V = list(
    train = asc_train + b_time * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED,
    Swissmetro = asc_sm + b_time * SM_TT_SCALED + b_cost * SM_COST_SCALED,
    car = asc_car + b_time * CAR_TT_SCALED + b_cost * CAR_CO_SCALED
  )
  
  mnl_settings = list(
    alternatives=c(train = 1, Swissmetro = 2, car = 3),
    avail = list(train = TRAIN_AV_SP, Swissmetro = SM_AV, car = CAR_AV_SP),
    choiceVar = CHOICE,
    utilities = V
  )
  
  P[["model"]] = apollo_mnl(mnl_settings, functionality)
  
  # panel multiplications should not matter in an MNL but it is in the Apollo examples
  P = apollo_panelProd(P, apollo_inputs, functionality)
  
  P = apollo_prepareProb(P, apollo_inputs, functionality)
  
  return(P)
}

model = apollo_estimate(apollo_beta, apollo_fixed, apollo_probabilities, apollo_inputs)

timer = toc()

print(paste("Model estimation time:", (timer$toc - timer$tic)[[1]], "seconds"))


