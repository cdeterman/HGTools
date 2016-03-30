library("HGTools")
context("Binary Model Functions")

gc()
data(infert, package="datasets")
set.seed(123)
bm <- suppressWarnings(as.big.matrix(infert))
fnn.model <- fast_neuralnet_bm(case~parity+induced+spontaneous, bm, stepmax=2000,
                            err.fct="ce", linear.output=FALSE, likelihood=TRUE)


# class(fnn.model) <- "list"
MODEL <- list(model.list = fnn.model$model.list, weights = fnn.model$weights, act.fct = fnn.model$act.fct, linear.output=fnn.model$linear.output)
covariate <- suppressWarnings(as.big.matrix(infert[,c("parity","induced","spontaneous")]))
out <- compute(x=MODEL, covariate=covariate, 
               model_type = "binary")
# str(out)
test_that("compute returns correct objects", {
  expect_true("dvs" %in% slotNames(out))
  expect_true("ivs" %in% slotNames(out))
  })

# local unit tests until I can create appropriate reproducible data
test_that("activate_scores_table works", {
  load(paste0(Sys.getenv("HOME"), '/Documents/HGmiscTools/tests/data/binary_result.rnn'))
  load(paste0(Sys.getenv("HOME"), '/Documents/HGmiscTools/tests/data/oa_knee_test.rda'))
  BT <- suppressWarnings(as.big.matrix(T))
  BC <- deepcopy(BT, MODEL$model.list$variables)
  RESULT <- compute(MODEL, BC, model_type="binary")
  out <- activation_scores_table(RESULT, BT, d_var = "has_oa_knee", step=0.01)
  expect_true(ncol(out) == 2, "activation_scores_table doesn't return 8 columns")
  expect_equivalent(out[,1], RESULT@net.result[,1])
})

test_that("demographic_profile works", {
    load(paste0(Sys.getenv("HOME"), '/Documents/HGmiscTools/tests/data/binary_result.rnn'))
    load(paste0(Sys.getenv("HOME"), '/Documents/HGmiscTools/tests/data/oa_knee_test.rda'))
    BT <- suppressWarnings(as.big.matrix(T))
    BC <- deepcopy(BT, MODEL$model.list$variables)
    RESULT <- compute(MODEL, BC, model_type="binary")
    scores <- assign_scores_and_categories(RESULT, BT, allowParallel=FALSE)
    demog_profile <- demographic_profile(scores, BT, "has_oa_knee", categories=FALSE)
#     expect_true(ncol(demog_profile) == 7, "all categories present")
})

