
#' @title Training of neural networks
#' @description This is a replication of the neuralnet package to be compatible with 
#' big.matrix objects. neuralnet is used to train neural networks using backpropagation, 
#' resilient backpropagation (RPROP) with (Riedmiller, 1994) or without weight backtracking 
#' (Riedmiller and Braun, 1993) or the modified globally convergent version (GRPROP) by 
#' Anastasiadis et al. (2005). The function allows flexible settings through custom-choice 
#' of error and activation function. Furthermore the calculation of generalized weights 
#' (Intrator O. and Intrator N., 1993) is implemented.
#' @param formula a symbolic description of the model to be fitted.
#' @param data a data frame, matrix, or big.matrix containing the variables specified in formula.
#' @param hidden a vector of integers specifying the number of hidden neurons (vertices) in each layer.
#' @param threshold a numeric value specifying the threshold for the partial derivatives of the error function as stopping criteria.
#' @param stepmax the maximum steps for the training of the neural network. Reaching this maximum leads to a stop of the neural network's training process.
#' @param rep the number of repetitions for the neural network's training.
#' @param startweights a vector containing starting values for the weights. The weights will not be randomly initialized.
#' @param learningrate.limit a vector or a list containing the lowest and highest limit for the learning rate. Used only for RPROP and GRPROP.
#' @param learningrate.factor a vector or a list containing the multiplication factors for the upper and lower learning rate. Used only for RPROP and GRPROP.
#' @param learningrate a numeric value specifying the learning rate used by traditional backpropagation. Used only for traditional backpropagation.
#' @param lifesign a string specifying how much the function will print during the calculation of the neural network. 'none', 'minimal' or 'full'.
#' @param lifesign.step an integer specifying the stepsize to print the minimal threshold in full lifesign mode.
#' @param algorithm a string containing the algorithm type to calculate the neural network. The following types are possible: 'backprop', 'rprop+', 'rprop-', 'sag', or 'slr'. 'backprop' refers to backpropagation, 'rprop+' and 'rprop-' refer to the resilient backpropagation with and without weight backtracking, while 'sag' and 'slr' induce the usage of the modified globally convergent algorithm (grprop). See Details for more information.
#' @param err.fct a differentiable function that is used for the calculation of the error. Alternatively, the strings 'sse' and 'ce' which stand for the sum of squared errors and the cross-entropy can be used.
#' @param act.fct a differentiable function that is used for smoothing the result of the cross product of the covariate or neurons and the weights. Additionally the strings, 'logistic' and 'tanh' are possible for the logistic function and tangent hyperbolicus.
#' @param linear.output logical. If act.fct should not be applied to the output neurons set linear output to TRUE, otherwise to FALSE.
#' @param exclude a vector or a matrix specifying the weights, that are excluded from the calculation. If given as a vector, the exact positions of the weights must be known. A matrix with n-rows and 3 columns will exclude n weights, where the first column stands for the layer, the second column for the input neuron and the third column for the output neuron of the weight.
#' @param constant.weights a vector specifying the values of the weights that are excluded from the training process and treated as fix.
#' @param likelihood logical. If the error function is equal to the negative log-likelihood function, the information criteria AIC and BIC will be calculated. Furthermore the usage of confidence.interval is meaningfull.
#' @details The globally convergent algorithm is based on the resilient backpropagation without weight backtracking and additionally modifies one learning rate, either the learningrate associated with the smallest absolute gradient (sag) or the smallest learningrate (slr) itself. The learning rates in the grprop algorithm are limited to the boundaries defined in learningrate.limit.
#' @return \code{neuralnet} returns an object of class nn. An object of class nn is a list 
#' containing at most the following components:
#' @return \item{call}{the matched call}
#' @return \item{response}{extracted from the data argument.}
#' @return \item{covariate}{the variables extracted from the data argument}
#' @return \item{model.list}{a list containing the covariates and the response variables extracted from the formula argument.}
#' @return \item{err.fct}{the error function.}
#' @return \item{act.fct}{the activation function.}
#' @return \item{data}{the data argument.}
#' @return \item{net.result}{a list containing the overall result of the neural network for every repetition.}
#' @return \item{weights}{a list containing the fitted weights of the neural network for every repetition.}
#' @return \item{generalized.weights}{a list containing the generalized weights of the neural network for every repetition.}
#' @return \item{result.matrix}{a matrix containing the reached threshold, needed steps, error, AIC and BIC (if computed) and weights for every repetition. Each column represents one repetition.}
#' @return \item{startweights}{a list containing the startweights of the neural network for every repetition.}
#' @author Stefan Fritsch, Frauke Guenther, Charles Determan Jr.
#' @references Riedmiller M. (1994) \emph{Rprop - Description and Implementation Details.} Technical Report. University of Karlsruhe.
#'
#' Riedmiller M. and Braun H. (1993) \emph{A direct adaptive method for faster backpropagation learning: The RPROP algorithm.} Proceedings of the IEEE International Conference on Neural Networks (ICNN), pages 586-591. San Francisco.
#' 
#' Anastasiadis A. et. al. (2005) \emph{New globally convergent training scheme based on the resilient propagation algorithm.} Neurocomputing 64, pages 253-270.
#' 
#' Intrator O. and Intrator N. (1993) \emph{Using Neural Nets for Interpretation of Nonlinear Models.} Proceedings of the Statistical Computing Section, 244-249 San Francisco: American Statistical Society (eds).
#' @name neuralnet
# ' @exportMethod neuralnet
#' @export
setGeneric("neuralnet", function(formula, data, ...){
  standardGeneric("neuralnet")
})


#' @title Computation of a given neural network for given covariate vectors
#' @description compute, a method for objects of class nn, typically produced by neuralnet. Computes
#' the outputs of all neurons for specific arbitrary covariate vectors given a trained neural network.
#' Please make sure that the order of the covariates is the same in the new matrix or dataframe as in
#' the original neural network.
#' @param x an object of class nn, or contains at least the \code{weights}, \code{act.fct}, and
#' \code{linear.output} components from \code{\link{neuralnet}} output.
#' @param covariate a dataframe or matrix containing the variables that had been used to train the neural network.
#' @param rep an integer indicating the neural network's repetition which should be used.
#' @param model_type State specific model type that was fit (e.g. payor)
#' @return \item{compute}{returns a list containing the following components:}
#' @return \item{neurons}{a list of the neurons' output for each layer of the neural network.}
#' @return \item{net.result}{a matrix containing the overall result of the neural network.}
#' @details This is a replication of the \link[neuralnet]{compute} function slightly modified for OOP
#' to accept \code{big.matrix} objects in addition to \code{matrix} or \code{data.frame}.
#' @name compute
#' @import assertive
# ' @exportMethod compute
#' @export
setGeneric("compute", valueClass = "nn_model", function(x, covariate,...){
  standardGeneric("compute")
})


#' @title Assign Testing Data Categories
#' @description Takes the columns containing category identificatiers (i.e. 0 or 1) in the
#' test dataset and returns a vector of the assigned category for the specific model type.
#' For example, payor model returns categories 'C','M','D', & 'S'.
#' @param object Object of class nn_model or matrix/big.matrix which contains the 
#' necessary columns for assigning categories
#' @return A character vector containing the assigned categories for the specific model
#' @name assign_categories
# ' @exportMethod assign_categories
#' @export
setGeneric("assign_categories", valueClass="character", function(object, ...){
  standardGeneric("assign_categories")
})


#' @title Assign Predicted Categories
#' @description This function takes the 'nn_model' object from the 
#' \code{\link{compute}} function.
#' and returns a vector of the specific category type according to the model type.
#' @param object A 'nn_model' or 'model_subset' object
#' @return A character vector representing the predicted model categories
#' @seealso \code{\link{assign_predicted_payor_category}}
#' @name assign_predicted_categories
#' @rdname assign_predicted_categories-methods
# ' @exportMethod assign_predicted_categories
#' @export
setGeneric("assign_predicted_categories", valueClass = "character", function(object) {
  standardGeneric("assign_predicted_categories")
})


#' @title Assign Scores and Confidence Categories
#' @description Wrapper to quickly report the confidence categories and
#' scaled scores for each sample from a neural net model.  This assigns scores
#' based on the raw quantile level.
#' @param object An 'nn_model' object
#' @param raw_x The raw data initially submitted to \code{\link{neuralnet}}
#' @param breaks Optional parameter to alter default quantile breaks
#' @param cutoffs Optional parameter to alter default category breaks
#' @param allowParallel Optional parameter to allow parallel processing
#' @return A 'model_scores' object which contains:
#' @return Matrix containing the four original category columns, raw activation values,
#' calculated scores, and confidence categories
#' @seealso \code{\link{assign_payor_scores_and_categories}}
#' @name assign_scores_and_categories
#' @rdname assign_scores_and_categories-methods
# ' @exportMethod assign_scores_and_categories
#' @export
setGeneric("assign_scores_and_categories", valueClass = "model_scores", 
           function(object, raw_x, ...){
             standardGeneric("assign_scores_and_categories")
})


#' @title Activation Scores Matrix
#' @description This function is designed to return a matrix
#' containing the activation levels and associated scores (i.e. p-values)
#' for a trained model.
#' @param object An "nn_model" object returned from \code{\link{compute}}
#' @param raw_x The raw data initially submitted to \code{\link{neuralnet}}
#' @param d_var A character vector of the dependent variable column
#' name(s).
#' @param step A fraction denoting the steps for the quantiles 
#' (e.g. 0.01 = 100 steps)
#' @param breaks Optional parameter to alter default quantile breaks
#' @param allowParallel Optional parameter to allow parallel processing
#' @return A matrix containing the activation levels from the "nn_model" and
#' the calculated scores (i.e. p-values).
#' @name activation_scores_table
#' @rdname activation_scores_table-methods
# ' @exportMethod activation_scores_table
#' @export
setGeneric("activation_scores_table", valueClass="matrix",
           function(object, raw_x, ...){
             standardGeneric("activation_scores_table")
           })

#' @title Subset Score Categories
#' @description This function subsets the output from \code{\link{assign_scores_and_categories}} 
#' to evaluate prediction metrics above a given cutoff (e.g. 4, 7, etc.)
#' @param object A 'model_scores' object
#' @param cutoff The cutoff for filtering predictions (e.g. all those with confidence category >= 5)
#' @name score_subset
#' @return A 'model_subset' object
#' @return \describe{
#'  \item{pred}{Subset of the neuralnet activations}
#'  \item{raw}{Subset of the test data category columns}
#'  }
#' @seealso \code{\link{score_subset,payor_scores-method}}
#' @rdname score_subset-methods
# ' @exportMethod score_subset
#' @export
setGeneric("score_subset", valueClass = "model_subset", 
           function(object, cutoff, ...){
             standardGeneric("score_subset")
})


#' @export
setGeneric("AUC",
           function(object, ...){
               standardGeneric("AUC")
           })

#' @title Probability Classification Profile
#' @description This function calculates the probability of the group
#' being accurately predicted across user defined bins of length \code{step}.
#' This provide a matrix of values that can be used to rapidly create a lift
#' curve with the \code{\link{lift_curve}} function.
#' @param x The output from \code{\link{assign_scores_and_categories}}
#' @param step The width of the bins created from 0 to 1
#' @param allowParallel Optional argument to utilize parallel processing
#' @export
setGeneric("percentile_profile", valueClass = "percentiles", 
           function(object, ...){
             standardGeneric("percentile_profile")
           })



#' @title Create Lift Curve for Classification Model
#' @description This function quickly creates the lift curves
#' for models corresponding to the total utilization with 
#' respect to score quantile levels.
#' @param x The output from \code{\link{percentile_profile}}
#' @param data A data.frame or matrix containing the original dependent
#' variable column
#' @param dvs A string specifying the dependent variable
#' @param filename The name of the file to be saved as a pdf
#' @importFrom reshape2 melt
#' @import ggplot2
#' @export
setGeneric("lift_curve",
           function(object, ...){
             standardGeneric("lift_curve")
           })

#' @title Create Classification Curve for Classification Model
#' @description This function quickly creates the classification curves
#' for models corresponding to the probability of correctly
#' classified categories with respect to quantile levels.
#' @param x The output from \code{\link{percentile_profile}}
#' @param filename The name of the file to be saved as a pdf
#' @importFrom reshape2 melt
#' @import ggplot2
#' @export
setGeneric("classification_curve",
           function(object, ...){
             standardGeneric("classification_curve")
           })

#' @title Create Gains Chart
#' @description This function quickly creates the cumulative classification 
#' curves for models corresponding to quantile levels.
#' @param x The output from \code{\link{percentile_profile}}
#' @param filename The name of the file to be saved as a pdf
#' @importFrom reshape2 melt
#' @import ggplot2
#' @export
setGeneric("gains_chart",
           function(object, ...){
             standardGeneric("gains_chart")
           })


#' @title Demographic Profile
#' @description This function is used for a naieve demographic profile
#' of each category within a scored and categorized model.  It returns
#' a matrix of size M x (7*num.groups).  The values within a simply the
#' sum of each column for each category within each group (as data is binary).
#' @param x A 'model_scores' object (i.e. the output from 
#' \code{\link{assign_scores_and_categories}})
#' @param data The raw data initially used to fit the model containing all variables.
#' @return Matrix containing the sums of each column within each category of
#' each group.
#' @note The 'data' parameter may also be a subset of columns which contains only
#' the variables of interest.  However, the subset must still contain the dependent
#' variables.  This is used as an internal check to make sure only data for the specific
#' model is evaluated.
#' @export
setGeneric("demographic_profile", valueClass = "matrix", 
           function(object, ...){
             standardGeneric("demographic_profile")
           })


#' @title Extract Breakpoints
#' @description This function is used to extract the breakpoints
#' for a standard 7 category scored model.  It returns a vector
#' of values starting from 0 and ending with 1.
#' @param x A 'model_scores' object (i.e. the output from 
#' \code{\link{assign_scores_and_categories}})
#' @return A numeric vector containing the breakpoints
#' @note The vector returned does not include the absolute min/max from the
#' scored data.  This was intentional in order to be more inclusive
#' when the model is used to score.
#' @export
setGeneric("extract_breakpoints", valueClass = "vector", 
           function(object, ...){
               standardGeneric("extract_breakpoints")
           })
