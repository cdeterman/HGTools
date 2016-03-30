setClassUnion("matrixORbigmatrix", c("matrix","big.matrix")) # remove when upgrade bigalgebra
setClassUnion("matrixORdataframe", c("matrix", "data.frame"))
setClassUnion("integerORnumericORmissing", c("integer", "numeric", "missing"))
setClassUnion("logicalORmissing", c("logical","missing"))

setOldClass("fnn")

setClass("nn_model", slots = c(neurons = "list", 
                               net.result = "matrixORbigmatrix",
                               dvs = "character",
                               ivs = "character"))

#' @export
setClass("nn_payor", contains = "nn_model")
#'@export
setClass("nn_binary", contains = "nn_model")

setClass("model_scores", slots=c(x = "matrix"))
setClass("payor_scores", contains = "model_scores")
setClass("binary_scores", contains = "model_scores")

setClass("model_subset", slots=c(raw="matrixORbigmatrix", pred="matrixORbigmatrix"))
setClass("payor_subset", contains = "model_subset")
setClass("binary_subset", contains = "model_subset")

setClass("percentiles", slots=c(percentiles="matrix"))
setClass("payor_percentiles", contains = "percentiles")
setClass("binary_percentiles", contains = "percentiles")