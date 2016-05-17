## ----setup, include=FALSE, cache=FALSE-----------------------------------
library(knitr)
opts_chunk$set(
concordance=TRUE
)

## ----install, eval=FALSE-------------------------------------------------
#  # install devtools and bigmemory backends
#  install.packages(c("devtools","bigmemory", "bigalgebra", "biganalytics"))
#  
#  # the other dependencies should be installed automatically
#  devtools::install_github("cdeterman/HGTools")

## ----gridSearch, eval = FALSE--------------------------------------------
#  denovo.grid(method = "neuralnet", res = 3)

## ----rfgrid, eval=FALSE--------------------------------------------------
#  # Note the dvs denotes the dependent variables to omit from the estimation
#  # This can be omitted but you will receive a warning
#  denovo.grid("neuralnet", res=5, data=trainingData, dvs=c("my_dv"))

## ----manualGrid, eval=FALSE----------------------------------------------
#  expand.grid(.hidden = seq(2,5), .threshold = c(5, 1))

## ----loadData, eval=FALSE------------------------------------------------
#  data("adhd_train")
#  data("adhd_test")

## ----trainExample, eval=FALSE--------------------------------------------
#  # To save space I am indexing the names
#  cnames <- colnames(training)
#  ivs <- cnames[3:ncol(training)]
#  dvs <- cnames(training)[1]
#  
#  f <- as.formula(paste(dvs, " ~ ", paste(ivs, collapse= "+")))
#  
#  fit_nn <- train(formula = f,
#                  data = training,
#                  testData = testing,
#                  method = "neuralnet",
#                  grid = grid,
#                  k = 5,
#                  metric = "AUC"
#  )

## ----doParallel, eval=FALSE----------------------------------------------
#  # register 8 cores
#  cl <- makeCluster(8)
#  registerDoParallel(8)
#  
#  # make sure to stop cluster when completed
#  stopCluster(cl)

## ----trainParallel, eval=FALSE-------------------------------------------
#  fit_nn <- train(formula = f,
#                  data = training,
#                  testData = testing,
#                  method = "neuralnet",
#                  grid = grid,
#                  k = 5,
#                  metric = "AUC",
#                  allowParallel = TRUE
#  )

