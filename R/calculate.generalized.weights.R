calculate.generalized.weights <-
    function (weights, neuron.deriv, net.result) 
    {
        for (w in 1:length(weights)) {
            weights[[w]] <- remove.intercept(weights[[w]])
        }
        generalized.weights <- NULL
        for (k in 1:ncol(net.result)) {
            for (w in length(weights):1) {
                if (w == length(weights)) {
                    temp <- neuron.deriv[[length(weights)]][, k] * 
                        1/(net.result[, k] * (1 - (net.result[, k])))
                    delta <- tcrossprod(temp, weights[[w]][, k])
                }
                else {
                    #                 delta <- tcrossprod(delta * neuron.deriv[[w]], 
                    #                   weights[[w]])
                    delta <- tcrossprod(delta * as.matrix(neuron.deriv[[w]]), 
                                        weights[[w]])
                }
            }
            generalized.weights <- cbind(generalized.weights, delta)
        }
        return(generalized.weights)
    }
