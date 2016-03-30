# library("HGTools")
# context("Payor Model Functions")
# 
# # global neuralnet objects
# # data(infert, package="datasets")
# # set.seed(123)
# # fnn.model <- fast_neuralnet(case~parity+induced+spontaneous, infert, stepmax=2000,
# #                             err.fct="ce", linear.output=FALSE, likelihood=TRUE)
# # set.seed(123)
# # nn.model <- neuralnet::neuralnet(case~parity+induced+spontaneous, infert, 
# #                                  err.fct="ce", linear.output=FALSE, likelihood=TRUE)
# # 
# # set.seed(123)
# # bm <- suppressWarnings(as.big.matrix(infert))
# # bm.nn.model <- big.neuralnet(case~parity+induced+spontaneous, bm, 
# #                              err.fct="ce", linear.output=FALSE, likelihood=TRUE)
# # bm.nn.net.result <-  bm.nn.model$net.result
# # bm.nn.weights <-  bm.nn.model$weights
# # bm.nn.startweights <-  bm.nn.model$startweights
# # bm.nn.generalized.weights <-  bm.nn.model$generalized.weights
# # bm.nn.result.matrix <-  bm.nn.model$result.matrix
# # rm(bm.nn.model)
# # gc()
# # 
# # set.seed(123)
# # bm.fnn.model <- fast_neuralnet_bm(case~parity+induced+spontaneous, bm, 
# #                                   err.fct="ce", linear.output=FALSE, likelihood=TRUE)
# # bm.fnn.net.result <-  bm.fnn.model$net.result
# # bm.fnn.weights <-  bm.fnn.model$weights
# # bm.fnn.startweights <-  bm.fnn.model$startweights
# # bm.fnn.generalized.weights <-  bm.fnn.model$generalized.weights
# # bm.fnn.result.matrix <-  bm.fnn.model$result.matrix
# # rm(bm.fnn.model)
# # gc()
# 
# test_that("assign_scores_and_categories payor model tests", {
#   raw <- cbind(replicate(4, sample(c(1,0), 100, replace=TRUE)))
#   colnames(raw) <- c("dv_had_c","dv_had_m","dv_had_d","dv_had_s")
#   net.result <- cbind(runif(100), runif(100), runif(100), runif(100))
#   # output from big.compute
#   tmp <- new("payor", neurons=list("place_holder"), net.result=net.result)
#   sc <- assign_scores_and_categories(tmp, raw)
#   
#   expect_is(sc, "payor_scores")
#   expect_that(sc@x[,1:4], is_identical_to(raw))
#   expect_equal(sc@x[,5:8], net.result, check.attributes=FALSE)
# })
# 
# test_that("assigning categories to payor model tests", {
#   tmp <- c("C", rep("M", 6), rep("S", 6), rep("D", 3))
#   pc <- c("1000","0100", "1100", "1111", "0111", "0101", "0110",
#           "0001", "0000", "0011", "1001", "1011", "1101",
#           "0010", "1010", "1110")
#   codes <- do.call("rbind", sapply(pc, strsplit, split=""))
#   raw <- cbind(replicate(4, sample(c(1,0), 100, replace=TRUE)))
#   colnames(raw) <- c("dv_had_c","dv_had_m","dv_had_d","dv_had_s")
#   net.result <- cbind(runif(100), runif(100), runif(100), runif(100))
#   
#   tmp_mod <- new("payor", neurons=list("place_holder"), net.result=net.result)
#   sc <- assign_scores_and_categories(tmp_mod, raw)
#   
#   # tests
#   expect_that(assign_categories(codes, model_type="payor"), is_identical_to(tmp))
# })
# 
# test_that("assign_payor_category requires specific inputs", {
#   sub_df <- new("payor_subset", 
#                 raw=replicate(4, sample(c(1,0), 100, replace=TRUE)),
#                 pred=replicate(4, runif(100)))
#   expect_error(assign_categories(sub_df@raw))
#   expect_error(assign_categories(raw))
# })
# 
# test_that("score_subset requires appropriate arguments for payor model", {
#   sc <- new("payor_scores", x=cbind(replicate(4, sample(c(1,0), 100, replace=TRUE)), 
#                                     replicate(4, runif(100)),
#                                     replicate(4, sample(seq(999), 100)),
#                                     replicate(4, sample(seq(7), 100, replace=TRUE))))
#   expect_error(score_subset(sc))
#   expect_error(score_subset(sc, 5, 'T'))
#   expect_error(score_subset(sc, matrix(seq(10))))
# })
# 
# test_that("assign_predicted_cateogy requires specific inputs", {
#   # 'payor' class
#   raw <- cbind(replicate(4, sample(c(1,0), 100, replace=TRUE)))
#   colnames(raw) <- c("dv_had_c","dv_had_m","dv_had_d","dv_had_s")
#   net.result <- cbind(runif(100), runif(100), runif(100), runif(100))
#   # output from big.compute
#   tmp <- new("payor", neurons=list("place_holder"), net.result=net.result)
#   
#   # payor_subset class
#   sub_df <- new("payor_subset", 
#                 raw=replicate(4, sample(c(1,0), 100, replace=TRUE)),
#                 pred=replicate(4, runif(100)))
#   
#   # tests
#   expect_error(assign_predicted_category(sub_df@pred))
#   expect_error(assign_predicted_category(tmp@net.result))
# })
# 
# # test percentile_profile
# 
# # local unit tests until I can create appropriate reproducible data
# test_that("activate_scores_table works", {
#   load(paste0(Sys.getenv("HOME"), '/Documents/HGTools/tests/data/payor_result.rnn'))
#   load(paste0(Sys.getenv("HOME"), '/Documents/HGTools/tests/data/test_std.rda'))
#   BT <- suppressWarnings(as.big.matrix(T))
#   out <- activation_scores_table(RESULT, BT, step=0.01)
#   expect_true(ncol(out) == 8, "activation_scores_table doesn't return 8 columns")
#   expect_equivalent(out[,1:4], RESULT@net.result[,1:4])
# })
