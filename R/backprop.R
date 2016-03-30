backprop <-
    function (gradients, weights, length.weights, nrow.weights, ncol.weights, 
              learningrate.bp, exclude) 
    {
        weights <- unlist(weights)
        if (!is.null(exclude)) 
            weights[-exclude] <- weights[-exclude] - gradients * 
            learningrate.bp
        else weights <- weights - gradients * learningrate.bp
        list(gradients.old = gradients, weights = relist(weights, 
                                                         nrow.weights, ncol.weights), learningrate = learningrate.bp)
    }
