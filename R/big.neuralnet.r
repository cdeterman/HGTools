
#' @title Training of neural networks
#' @description This is a replication of the \code{neuralnet} package to be compatible with 
#' big.matrix objects.  
#' \code{neuralnet} is used to train neural networks using backpropagation, 
#' resilient backpropagation (RPROP) with (Riedmiller, 1994) or without weight backtracking 
#' (Riedmiller and Braun, 1993) or the modified globally convergent version (GRPROP) by 
#' Anastasiadis et al. (2005). The function allows flexible settings through custom-choice 
#' of error and activation function. Furthermore the calculation of generalized 
#' weights (Intrator O. and Intrator N., 1993) is implemented.
#' @param formula a symbolic description of the model to be fitted.
#' @param data a data frame (or big.matrix) containing the variables specified in formula.
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
#' @return neuralnet returns an object of class nn. An object of class nn is a list containing at most the following components:
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
#' @import bigalgebra
#' @export


big.neuralnet <-
function (formula, 
          data, 
          hidden = 1, 
          threshold = 0.01, 
          stepmax = 1e+05, 
          rep = 1, 
          startweights = NULL, 
          learningrate.limit = NULL, 
          learningrate.factor = list(minus = 0.5, plus = 1.2), 
          learningrate = NULL, 
          lifesign = "none", 
          lifesign.step = 1000, 
          algorithm = "rprop+", 
          err.fct = "sse", 
          act.fct = "logistic", 
          linear.output = TRUE, 
          exclude = NULL, 
          constant.weights = NULL, 
          likelihood = FALSE,
          low_size = TRUE) 
{
    call <- match.call()
    options(scipen = 100, digits = 10)
    
    # verify inputs are appropriate
    result <- varify.variables(data, formula, startweights, learningrate.limit, 
                               learningrate.factor, learningrate, lifesign, algorithm, 
                               threshold, lifesign.step, hidden, rep, stepmax, err.fct, 
                               act.fct)
    data <- result$data
    formula <- result$formula
    startweights <- result$startweights
    learningrate.limit <- result$learningrate.limit
    learningrate.factor <- result$learningrate.factor
    learningrate.bp <- result$learningrate.bp
    lifesign <- result$lifesign
    algorithm <- result$algorithm
    threshold <- result$threshold
    lifesign.step <- result$lifesign.step
    hidden <- result$hidden
    rep <- result$rep
    stepmax <- result$stepmax
    model.list <- result$model.list
    
    # empty objects to fill
    matrix <- NULL
    list.result <- NULL
    
    # generate initial variables
    result <- generate.initial.variables(data, model.list, hidden, 
                                         act.fct, err.fct, algorithm, linear.output, formula)
    covariate <- result$covariate
    response <- result$response
    err.fct <- result$err.fct
    err.deriv.fct <- result$err.deriv.fct
    act.fct <- result$act.fct
    act.deriv.fct <- result$act.deriv.fct
    
    
    for (i in 1:rep) {
        if (lifesign != "none") {
            lifesign <- display(hidden, threshold, rep, i, lifesign)
        }
        flush.console()
        
        # calculate neuralnet scores
        result <- 
          calculate.neuralnet(learningrate.limit = learningrate.limit, 
                              learningrate.factor = learningrate.factor, 
                              covariate = covariate, #big.matrix
                              response = response,# big.matrix
                              data = data, # big.matrix
                              model.list = model.list, 
                              threshold = threshold, lifesign.step = lifesign.step, 
                              stepmax = stepmax, hidden = hidden, lifesign = lifesign, 
                              startweights = startweights, algorithm = algorithm, 
                              err.fct = err.fct, err.deriv.fct = err.deriv.fct, 
                              act.fct = act.fct, act.deriv.fct = act.deriv.fct, 
                              rep = i, linear.output = linear.output, exclude = exclude, 
                              constant.weights = constant.weights, likelihood = likelihood, 
                              learningrate.bp = learningrate.bp)
        # add results to list object
        if (!is.null(result$output.vector)) {
            list.result <- c(list.result, list(result))
            matrix <- cbind(matrix, result$output.vector)
        }
    }
    
    flush.console()
    if (!is.null(matrix)) {
        weight.count <- length(unlist(list.result[[1]]$weights)) - 
            length(exclude) + length(constant.weights) - sum(constant.weights == 
            0)
        if (!is.null(startweights) && length(startweights) < 
            (rep * weight.count)) {
            warning("some weights were randomly generated, because 'startweights' did not contain enough values", 
                call. = F)
        }
        ncol.matrix <- ncol(matrix)
    }else{
      ncol.matrix <- 0
    }
    
    if(ncol.matrix < rep){warning(sprintf("algorithm did not converge in %s of %s repetition(s) within the stepmax", 
                                            (rep - ncol.matrix), rep), call. = FALSE)}
    
    # generate formatted output
    nn <- generate.output(covariate, call, rep, threshold, matrix, 
                          startweights, model.list, response, err.fct, act.fct, 
                          data, list.result, linear.output, exclude, low_size)
    return(nn)
}
