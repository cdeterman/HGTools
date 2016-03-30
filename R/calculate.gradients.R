# neurons = result$neurons, # list of big.matrices
# neuron.deriv = result$neuron.deriv, # [[1]] is big.matrix
# err.deriv = err.deriv, # big.matrix

#' @importMethodsFrom bigmemory as.matrix
#' @importMethodsFrom biganalytics apply
calculate.gradients <-
    function (weights, length.weights, neurons, neuron.deriv, err.deriv, 
              exclude, linear.output) 
    {
        #err.deriv <- err.deriv[,]
        neurons <- lapply(neurons, function(x) x[,])
        neuron.deriv[[1]] <- as.matrix(neuron.deriv[[1]])
        # need is.na equation for big.matrices
        if (any(is.na(err.deriv))) {
            stop("the error derivative contains a NA; varify that the derivative function does not divide by 0 (e.g. cross entropy)", 
                 call. = FALSE)
        }
        
        if (!linear.output) {
            delta <- neuron.deriv[[length.weights]] * err.deriv
        }else{
            delta <- err.deriv
        } 
        
        ###############################
        # current place of performance hit
        ###############################
        gradients <- crossprod(neurons[[length.weights]], delta)
        #gradients <- as.matrix(transposeBM(neurons[[length.weights]]) %*% delta)
        if (length.weights > 1) 
            for (w in (length.weights - 1):1) {
                
                # need tcrossprod for big.matrix objects
                delta <- neuron.deriv[[w]] * tcrossprod(delta, remove.intercept(weights[[w + 1]]))
                #delta <- neuron.deriv[[w]] * (delta %*% transposeBM(remove.intercept(weights[[w + 1]])))
                
                gradients <- c(crossprod(neurons[[w]], delta), gradients)
                
                # To get a nice vector, need to convert to matrix
                # Smaller so should be minimal performance hit
                # Still major bottleneck point, getting rid of as.matrix would be best
                # need 'c' function for big.matrix objects
                
                #gradients <- c(as.matrix(transposeBM(neurons[[w]]) %*% delta), gradients)
                #gradients <- c(apply(transposeBM(neurons[[w]]) %*% delta, 2, FUN=function(x) x), gradients)
            }
        gradients[-exclude]
    }
