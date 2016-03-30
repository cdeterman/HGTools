rprop <-
    function (weights, 
              response, # big.matrix
              covariate, # big.matrix
              threshold, learningrate.limit, 
              learningrate.factor, stepmax, lifesign, lifesign.step, act.fct, 
              act.deriv.fct, err.fct, err.deriv.fct, algorithm, linear.output, 
              exclude, learningrate.bp) 
    {
        step <- 1
        nchar.stepmax <- max(nchar(stepmax), 7)
        length.weights <- length(weights)
        nrow.weights <- sapply(weights, nrow)
        ncol.weights <- sapply(weights, ncol)
        length.unlist <- length(unlist(weights)) - length(exclude)
        learningrate <- as.vector(matrix(0.1, nrow = 1, ncol = length.unlist))
        gradients.old <- as.vector(matrix(0, nrow = 1, ncol = length.unlist))
        if (is.null(exclude)) {
            exclude <- length(unlist(weights)) + 1
        }
        if (type(act.fct) == "tanh" || type(act.fct) == "logistic") {
            special <- TRUE
        }else{special <- FALSE} 
        
        if (linear.output) {
            output.act.fct <- function(x) {
                x
            }
            output.act.deriv.fct <- function(x) {
                matrix(1, nrow(x), ncol(x))
            }
        }else{
            if (type(err.fct) == "ce" && type(act.fct) == "logistic") {
                err.deriv.fct <- function(x, y) {
                    if(class(y) == 'big.matrix'){
                        x * (1 - y[,]) - y[,] * (1 - x)
                    }else{
                        x * (1 - y) - y * (1 - x)
                    }
                }
                linear.output <- TRUE
            }
            output.act.fct <- act.fct
            output.act.deriv.fct <- act.deriv.fct
        }
        
        # compute the neuralnet
        result <- compute.net(weights, length.weights, 
                              covariate = covariate, # big.matrix
                              act.fct = act.fct, 
                              act.deriv.fct = act.deriv.fct, 
                              output.act.fct = output.act.fct, 
                              output.act.deriv.fct = output.act.deriv.fct, 
                              special)
        
        #     print(class(result$net.result))
        
        
        #err.deriv <- err.deriv.fct(result$net.result, response)
        err.deriv <- err.deriv.fct(result$net.result, response[,])
        gradients <- calculate.gradients(weights = weights, 
                                         length.weights = length.weights, 
                                         neurons = result$neurons, # list of big.matrices
                                         neuron.deriv = result$neuron.deriv, # [[1]] is big.matrix
                                         err.deriv = err.deriv, # big.matrix
                                         exclude = exclude, 
                                         linear.output = linear.output)
        
        reached.threshold <- max(abs(gradients))
        min.reached.threshold <- reached.threshold
        while (step < stepmax && reached.threshold > threshold) {
            if (!is.character(lifesign) && step%%lifesign.step == 
                    0) {
                text <- paste("%", nchar.stepmax, "s", sep = "")
                cat(sprintf(eval(expression(text)), step), "\tmin thresh: ", 
                    min.reached.threshold, "\n", rep(" ", lifesign), 
                    sep = "")
                flush.console()
            }
            
            # covert to a switch statement
            if (algorithm == "rprop+") {
                result <- plus(gradients, gradients.old, weights, 
                               nrow.weights, ncol.weights, learningrate, learningrate.factor, 
                               learningrate.limit, exclude)
            }else{
                if (algorithm == "backprop") {
                    result <- backprop(gradients, weights, length.weights, 
                                       nrow.weights, ncol.weights, learningrate.bp, 
                                       exclude)
                }else{
                    result <- minus(gradients, gradients.old, weights, 
                                    length.weights, nrow.weights, ncol.weights, learningrate, 
                                    learningrate.factor, learningrate.limit, algorithm, 
                                    exclude)
                } 
            } 
            
            gradients.old <- result$gradients.old
            weights <- result$weights
            learningrate <- result$learningrate
            result <- compute.net(weights, length.weights, 
                                  covariate = covariate, # big.matrix
                                  act.fct = act.fct, act.deriv.fct = act.deriv.fct, 
                                  output.act.fct = output.act.fct, output.act.deriv.fct = output.act.deriv.fct, 
                                  special)
            err.deriv <- err.deriv.fct(result$net.result, response)
            gradients <- calculate.gradients(weights = weights, length.weights = length.weights, 
                                             neurons = result$neurons, neuron.deriv = result$neuron.deriv, 
                                             err.deriv = err.deriv, exclude = exclude, linear.output = linear.output)
            reached.threshold <- max(abs(gradients))
            if (reached.threshold < min.reached.threshold) {
                min.reached.threshold <- reached.threshold
            }
            step <- step + 1
        }
        if (lifesign != "none" && reached.threshold > threshold) {
            cat("stepmax\tmin thresh: ", min.reached.threshold, "\n", 
                sep = "")
        }
        return(list(weights = weights, step = as.integer(step), reached.threshold = reached.threshold, 
                    net.result = result$net.result, neuron.deriv = result$neuron.deriv))
    }
