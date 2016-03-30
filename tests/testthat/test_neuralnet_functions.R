library("HGTools")
context("Neural Net Functions")

# global neural net objects
data(infert, package="datasets")
bm <- suppressWarnings(as.big.matrix(infert))
fun <- function(x){
  1/(1+exp(-x))
}

# default models
set.seed(123)
fnn.model <- fast_neuralnet(case~parity+induced+spontaneous, infert, stepmax=2000,
                              err.fct="ce", linear.output=FALSE, likelihood=TRUE)

set.seed(123)
nn.model <- neuralnet::neuralnet(case~parity+induced+spontaneous, infert, 
                                err.fct="ce", linear.output=FALSE, likelihood=TRUE)

set.seed(123)
bm.nn.model <- big.neuralnet(case~parity+induced+spontaneous, bm, 
                                err.fct="ce", linear.output=FALSE, likelihood=TRUE)

set.seed(123)
bm.fnn.model <- fast_neuralnet_bm(case~parity+induced+spontaneous, bm, 
                               err.fct="ce", linear.output=FALSE, likelihood=TRUE)

# false likelihood models
set.seed(123)
fnn.model_false_likelihood <- 
  fast_neuralnet(case~parity+induced+spontaneous, 
                 infert, stepmax=2000,
                 err.fct="ce", linear.output=FALSE, 
                 likelihood=FALSE)

set.seed(123)
nn.model_false_likelihood <- 
  neuralnet::neuralnet(
    case~parity+induced+spontaneous, 
    infert, stepmax=2000, 
    err.fct="ce", linear.output=FALSE, likelihood=FALSE)

set.seed(123)
bm.nn.model_false_likelihood <- 
  big.neuralnet(case~parity+induced+spontaneous, 
                bm, stepmax=2000,
                err.fct="ce", linear.output=FALSE, likelihood=FALSE)

set.seed(123)
bm.fnn.model_false_likelihood <- 
  fast_neuralnet_bm(case~parity+induced+spontaneous, 
                    bm, stepmax=2000,
                    err.fct="ce", linear.output=FALSE, likelihood=FALSE)


# multiple hidden nodes
set.seed(123)
fnn.model_hidden <- 
  fast_neuralnet(case~parity+induced+spontaneous, 
                 infert, hidden=5,
                 err.fct="ce", linear.output=FALSE, 
                 likelihood=FALSE)

set.seed(123)
nn.model_hidden <- 
  neuralnet::neuralnet(
    case~parity+induced+spontaneous, 
    infert, hidden=5, 
    err.fct="ce", linear.output=FALSE, likelihood=FALSE)


set.seed(123)
bm.nn.model_hidden <- 
  big.neuralnet(case~parity+induced+spontaneous, 
                bm, hidden=5,
                err.fct="ce", linear.output=FALSE, likelihood=FALSE)


set.seed(123)
bm.fnn.model_hidden <- 
  fast_neuralnet_bm(case~parity+induced+spontaneous, 
                    bm, hidden=5,
                    err.fct="ce", linear.output=FALSE, likelihood=FALSE)

# custom function objects
# Note, the Rcpp functions require strings
set.seed(123)
nn.default.model_custom <- 
  neuralnet::neuralnet(
    case~parity+induced+spontaneous, 
    infert, act.fct = fun,
    err.fct="ce", linear.output=FALSE, likelihood=FALSE)

set.seed(123)
nn.model_custom <- 
  neuralnet(
    case~parity+induced+spontaneous, 
    infert, act.fct = fun,
    err.fct="ce", linear.output=FALSE, likelihood=FALSE)

set.seed(123)
bm.nn.model_custom <- 
  big.neuralnet(
    case~parity+induced+spontaneous, 
    bm, act.fct = fun,
    err.fct="ce", linear.output=FALSE, likelihood=FALSE)


# global compute objects
nn.compute <- neuralnet::compute(
  nn.model, infert[,c("parity","induced","spontaneous")])
fnn.compute <- fast_compute(
  fnn.model, infert[,c("parity","induced","spontaneous")])

test_that("neural nets return correct classes", {
  expect_that(fnn.model, is_a("fnn"))
  expect_that(nn.model, is_a("nn"))
  expect_that(bm.nn.model, is_a("nn"))
})

test_that("Rcpp neuralnet equivalent to default neuralnet", {
  expect_equivalent(nn.model$net.result, fnn.model$net.result)
  expect_equivalent(nn.model$weights, fnn.model$weights)
  expect_equivalent(nn.model$startweights, fnn.model$startweights)
  expect_equivalent(nn.model$generalized.weights, fnn.model$generalized.weights)
  expect_equivalent(nn.model$result.matrix, fnn.model$result.matrix)
})

test_that("Big matrix neuralnet equivalent to default neuralnet", {
  expect_equivalent(nn.model$net.result, bm.nn.model$net.result)
  expect_equivalent(nn.model$weights, bm.nn.model$weights)
  expect_equivalent(nn.model$startweights, bm.nn.model$startweights)
  expect_equivalent(nn.model$generalized.weights, bm.nn.model$generalized.weights)
  expect_equivalent(nn.model$result.matrix, bm.nn.model$result.matrix) 
})

test_that("Rcpp BigMatrix neuralnet equivalent to default neuralnet", {
  expect_equivalent(nn.model$net.result, bm.fnn.model$net.result)
  expect_equivalent(nn.model$weights, bm.fnn.model$weights)
  expect_equivalent(nn.model$startweights, bm.fnn.model$startweights)
  expect_equivalent(nn.model$generalized.weights, bm.fnn.model$generalized.weights)
  expect_equivalent(nn.model$result.matrix, bm.fnn.model$result.matrix) 
})

test_that("Rcpp compute equivalent to default compute", {
  expect_equivalent(nn.compute$net.result, fnn.compute$net.result)
  expect_equivalent(nn.compute$neurons, fnn.compute$neurons)
})

test_that("Rcpp neuralnet likeihood=FALSE equivalent to default neuralnet", {
  expect_equivalent(nn.model_false_likelihood$net.result, 
                    fnn.model_false_likelihood$net.result)
  expect_equivalent(nn.model_false_likelihood$weights, 
                    fnn.model_false_likelihood$weights)
  expect_equivalent(nn.model_false_likelihood$startweights, 
                    fnn.model_false_likelihood$startweights)
  expect_equivalent(nn.model_false_likelihood$generalized.weights, 
                    fnn.model_false_likelihood$generalized.weights)
  expect_equivalent(nn.model_false_likelihood$result.matrix, 
                    fnn.model_false_likelihood$result.matrix)
})

test_that("Big matrix neuralnet likelihood=FALSE equivalent to 
          default neuralnet", {
  expect_equivalent(nn.model_false_likelihood$net.result, 
                    bm.nn.model_false_likelihood$net.result)
  expect_equivalent(nn.model_false_likelihood$weights, 
                    bm.nn.model_false_likelihood$weights)
  expect_equivalent(nn.model_false_likelihood$startweights, 
                    bm.nn.model_false_likelihood$startweights)
  expect_equivalent(nn.model_false_likelihood$generalized.weights, 
                    bm.nn.model_false_likelihood$generalized.weights)
  expect_equivalent(nn.model_false_likelihood$result.matrix, 
                    bm.nn.model_false_likelihood$result.matrix) 
})

test_that("Rcpp BigMatrix neuralnet likelhood=FALSE equivalent to 
          default neuralnet", {
  expect_equivalent(nn.model_false_likelihood$net.result, 
                    bm.fnn.model_false_likelihood$net.result)
  expect_equivalent(nn.model_false_likelihood$weights,
                    bm.fnn.model_false_likelihood$weights)
  expect_equivalent(nn.model_false_likelihood$startweights, 
                    bm.fnn.model_false_likelihood$startweights)
  expect_equivalent(nn.model_false_likelihood$generalized.weights, 
                    bm.fnn.model_false_likelihood$generalized.weights)
  expect_equivalent(nn.model_false_likelihood$result.matrix, 
                    bm.fnn.model_false_likelihood$result.matrix) 
})


test_that("Rcpp neuralnet multiple hidden equivalent to default neuralnet", {
  expect_equivalent(nn.model_hidden$net.result, 
                    fnn.model_hidden$net.result)
  expect_equivalent(nn.model_hidden$weights, 
                    fnn.model_hidden$weights)
  expect_equivalent(nn.model_hidden$startweights, 
                    fnn.model_hidden$startweights)
  expect_equivalent(nn.model_hidden$generalized.weights, 
                    fnn.model_hidden$generalized.weights)
  expect_equivalent(nn.model_hidden$result.matrix, 
                    fnn.model_hidden$result.matrix)
})

test_that("Big matrix neuralnet multiple hidden equivalent to 
          default neuralnet", {
            expect_equivalent(nn.model_hidden$net.result, 
                              bm.nn.model_hidden$net.result)
            expect_equivalent(nn.model_hidden$weights, 
                              bm.nn.model_hidden$weights)
            expect_equivalent(nn.model_hidden$startweights, 
                              bm.nn.model_hidden$startweights)
            expect_equivalent(nn.model_hidden$generalized.weights, 
                              bm.nn.model_hidden$generalized.weights)
            expect_equivalent(nn.model_hidden$result.matrix, 
                              bm.nn.model_hidden$result.matrix) 
            })

test_that("Rcpp BigMatrix neuralnet multiple hidden equivalent to 
          default neuralnet", {
            expect_equivalent(nn.model_hidden$net.result, 
                              bm.fnn.model_hidden$net.result)
            expect_equivalent(nn.model_hidden$weights,
                              bm.fnn.model_hidden$weights)
            expect_equivalent(nn.model_hidden$startweights, 
                              bm.fnn.model_hidden$startweights)
            expect_equivalent(nn.model_hidden$generalized.weights, 
                              bm.fnn.model_hidden$generalized.weights)
            expect_equivalent(nn.model_hidden$result.matrix, 
                              bm.fnn.model_hidden$result.matrix) 
          })

test_that("Generic neuralnet custom act.fct equivalent to default neuralnet", {
  expect_equivalent(nn.model_custom$net.result, 
                    nn.default.model_custom$net.result)
  expect_equivalent(nn.model_custom$weights, 
                    nn.default.model_custom$weights)
  expect_equivalent(nn.model_custom$startweights, 
                    nn.default.model_custom$startweights)
  expect_equivalent(nn.model_custom$generalized.weights, 
                    nn.default.model_custom$generalized.weights)
  expect_equivalent(nn.model_custom$result.matrix, 
                    nn.default.model_custom$result.matrix)
})

test_that("Big matrix neuralnet custom act.fct equivalent to 
          default neuralnet", {
            expect_equivalent(nn.default.model_custom$net.result, 
                              bm.nn.model_custom$net.result)
            expect_equivalent(nn.default.model_custom$weights, 
                              bm.nn.model_custom$weights)
            expect_equivalent(nn.default.model_custom$startweights, 
                              bm.nn.model_custom$startweights)
            expect_equivalent(nn.default.model_custom$generalized.weights, 
                              bm.nn.model_custom$generalized.weights)
            expect_equivalent(nn.default.model_custom$result.matrix, 
                              bm.nn.model_custom$result.matrix) 
            })

test_that("neuralnet crashes with NA present", {
  infert[1,1] = NA
  bm[1,1] = NA
  
  expect_error(
    fast_neuralnet(case~parity+induced+spontaneous, infert,
                   err.fct="ce", linear.output=FALSE, likelihood=TRUE))
  expect_error(
    big.neuralnet(case~parity+induced+spontaneous, bm, 
                  err.fct="ce", linear.output=FALSE, likelihood=TRUE))
  expect_error(
    fast_neuralnet_bm(case~parity+induced+spontaneous, bm, 
                  err.fct="ce", linear.output=FALSE, likelihood=TRUE))
})
