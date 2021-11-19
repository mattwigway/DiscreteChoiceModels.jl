library(stringr)

benchmark = function (file, func=median, invocations=100) {
  times = numeric(invocations)
  
  for (i in 1:invocations) {
    output = system2(file.path(R.home("bin"), "R"), c("--no-save", "-f", file), stdout=T, stderr=F, wait=T)
    match = str_match(output, "Model estimation time: ([0-9\\.]+) seconds")
    stopifnot(sum(is.na(match[,2])) != 1)
    timestr = match[,2][!is.na(match[,2])][[1]]
    times[i] = as.double(timestr)
  }
  
  val = func(times)
  print(paste0(file, ": ", val, " seconds (", invocations, " invocations)"))
  return(val)
}

benchmark("biogeme_swissmetro.R")