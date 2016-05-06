
calculate.neuralnet <-
    function(data,      # big.matrix
             model.list, 
             hidden, 
             stepmax, 
             rep, 
             threshold, 
             learningrate.limit, 
             learningrate.factor, 
             lifesign, 
             covariate, # big.matrix
             response,  # big.matrix
             lifesign.step, 
             startweights, 
             algorithm, 
             act.fct, 
             act.deriv.fct, 
             err.fct, 
             err.deriv.fct, 
             linear.output, 
             likelihood, 
             exclude, 
             constant.weights, 
             learningrate.bp,
             dropout,
             visible_dropout,
             hidden_dropout) 
{
        # get starting time
        time.start.local <- Sys.time()
        
        # generate initial start weights
        result <- generate.startweights(model.list, hidden, startweights, 
                                        rep, exclude, constant.weights)
        weights <- result$weights
        exclude <- result$exclude
        nrow.weights <- sapply(weights, nrow)
        ncol.weights <- sapply(weights, ncol)
        
        result <- rprop(weights = weights, threshold = threshold, 
                        response = response, # big.matrix
                        covariate = covariate, # big.matrix
                        learningrate.limit = learningrate.limit, 
                        learningrate.factor = learningrate.factor, stepmax = stepmax, 
                        lifesign = lifesign, lifesign.step = lifesign.step, act.fct = act.fct, 
                        act.deriv.fct = act.deriv.fct, err.fct = err.fct, err.deriv.fct = err.deriv.fct, 
                        algorithm = algorithm, linear.output = linear.output, 
                        exclude = exclude, learningrate.bp = learningrate.bp,
                        dropout = dropout, visible_dropout = visible_dropout,
                        hidden_dropout = hidden_dropout)
        startweights <- weights
        weights <- result$weights
        step <- result$step
        reached.threshold <- result$reached.threshold
        net.result <- result$net.result
        
        
        error <- sum(err.fct(net.result, response))
        if (is.na(error) & type(err.fct) == "ce") {
            if (all(net.result <= 1, net.result >= 0)) {
                error <- sum(err.fct(net.result, response), na.rm = T)        
            }
        }
        
        if (!is.null(constant.weights) && any(constant.weights != 
                                                  0)) {
            exclude <- exclude[-which(constant.weights != 0)]
            
        }
        if (length(exclude) == 0) {exclude <- NULL}
        aic <- NULL
        bic <- NULL
        if (likelihood) {
            synapse.count <- length(unlist(weights)) - length(exclude)
            aic <- 2 * error + (2 * synapse.count)
            bic <- 2 * error + log(nrow(response)) * synapse.count
        }
        if (is.na(error)) {
            warning("'err.fct' does not fit 'data' or 'act.fct'", 
                    call. = F)      
        }
        
        # If user wants verbose output that processes are running
        if (lifesign != "none") {
            if (reached.threshold <= threshold) {
                cat(rep(" ", (max(nchar(stepmax), nchar("stepmax")) - 
                                  nchar(step))), step, sep = "")
                cat("\terror: ", round(error, 5), rep(" ", 6 - (nchar(round(error, 
                                                                            5)) - nchar(round(error, 0)))), sep = "")
                if (!is.null(aic)) {
                    cat("\taic: ", round(aic, 5), rep(" ", 6 - (nchar(round(aic, 
                                                                            5)) - nchar(round(aic, 0)))), sep = "")
                }
                if (!is.null(bic)) {
                    cat("\tbic: ", round(bic, 5), rep(" ", 6 - (nchar(round(bic, 
                                                                            5)) - nchar(round(bic, 0)))), sep = "")
                }
                time <- difftime(Sys.time(), time.start.local)
                cat("\ttime: ", round(time, 2), " ", attr(time, "units"), 
                    sep = "")
                cat("\n")
            }
        }
        
        
        if (reached.threshold > threshold){
            return(result = list(output.vector = NULL, weights = NULL))
        }
        output.vector <- c(error = error, reached.threshold = reached.threshold, 
                           steps = step)
        if (!is.null(aic)) {
            output.vector <- c(output.vector, aic = aic)
        }
        if (!is.null(bic)) {
            output.vector <- c(output.vector, bic = bic)
        }
        
        for (w in 1:length(weights)) output.vector <- c(output.vector, 
                                                        as.vector(weights[[w]]))
        
        generalized.weights <- 
            calculate.generalized.weights(weights, neuron.deriv = result$neuron.deriv, net.result = net.result)
        
        startweights <- unlist(startweights)
        weights <- unlist(weights)
        if (!is.null(exclude)) {
            startweights[exclude] <- NA
            weights[exclude] <- NA
        }
        
        startweights <- relist(startweights, nrow.weights, ncol.weights)
        weights <- relist(weights, nrow.weights, ncol.weights)
        return(
            list(generalized.weights = generalized.weights, weights = weights, 
                 startweights = startweights, net.result = result$net.result, 
                 output.vector = output.vector))
    }
