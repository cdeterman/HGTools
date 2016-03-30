#' @export
fast_compute <-
  function (x, covariate, rep = 1) 
  {    
    if(class(x) != "fnn"){
      assert_is_character(x$act.fct)
    }
    
    # check for NA
    weights <- x$weights[[rep]]
    nrow.weights <- sapply(weights, nrow)
    ncol.weights <- sapply(weights, ncol)
    init.weights <- as.relistable(weights)
    weights <- unlist(init.weights, recursive=T, use.names=T)
    if (any(is.na(weights))){
      weights[is.na(weights)] <- 0
    }
    x$weights <- relist(weights, nrow.weights, ncol.weights)
    
    if(is.big.matrix(covariate)){
      result <- c_compute_bm(x, covariate@address)
    }else{
      result <- c_compute(x, as.matrix(covariate))
    }
#     out <- new(model_type, 
#                neurons=result$neurons, net.result=result$net.result)
    
    return(list(neurons=result$neurons, net.result=result$net.result))
  }