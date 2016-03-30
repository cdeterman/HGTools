
#' @title "big.matrix" compatible computation of a given neural network for given covariate vectors
#' @description compute, a method for objects of class nn, typically produced by neuralnet. Computes
#' the outputs of all neurons for specific arbitrary covariate vectors given a trained neural network.
#' Please make sure that the order of the covariates is the same in the new matrix or dataframe as in
#' the original neural network.
#' @param x an object of class nn.
#' @param covariate a dataframe or matrix containing the variables that had been used to train the neural network.
#' @param rep an integer indicating the neural network's repetition which should be used.
#' @param model_type State specific model type that was fit (e.g. payor)
#' @return \item{compute}{returns a list containing the following components:}
#' @return \item{neurons}{a list of the neurons' output for each layer of the neural network.}
#' @return \item{net.result}{a matrix containing the overall result of the neural network.}
#' @details This is a replication of the \link[neuralnet]{compute} function only made compatible with big.matrix objects.
#' @import bigmemoryExt biganalytics
#' @export
big.compute <-
  function (x, covariate, rep = 1, model_type=NULL) 
  {
    assert_is_not_null(model_type)
    if(!model_type %in% model_types()[,1]){
      stop("Error: 'model_type' not defined")
    }
    nn <- x
    linear.output <- nn$linear.output
    weights <- nn$weights[[rep]]
    nrow.weights <- sapply(weights, nrow)
    ncol.weights <- sapply(weights, ncol)
    init.weights <- as.relistable(weights)
    weights <- unlist(init.weights, recursive=T, use.names=T)
    if (any(is.na(weights))){
      weights[is.na(weights)] <- 0
    }
    weights <- relist(weights, nrow.weights, ncol.weights)
    length.weights <- length(weights)
    
    # Need to be able to bind columns to big.matrix
    covariate <- cbindBM(covariate, 1, "left")

    act.fct <- nn$act.fct
    neurons <- list(covariate)
    if (length.weights > 1) 
      for (i in 1:(length.weights - 1)) {
        temp <- neurons[[i]] %*% weights[[i]]
        act.temp <- act.fct(temp[,])
        neurons[[i + 1]] <- cbind(1, act.temp)
      }
    temp <- neurons[[length.weights]] %*% weights[[length.weights]]
    if (linear.output) {
      net.result <- temp
    }else{
      net.result <- act.fct(temp[,])
    } 
    
    if(is.vector(net.result)){
      net.result <- as.matrix(net.result)
    }
    
    #out <- list(neurons = neurons, net.result = net.result)
    
    # neurons are the IV's
    # net.result is the predictions
    out <- list(neurons=neurons, net.result=net.result)

    return(out)
  }
