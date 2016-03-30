plus <-
    function (gradients, gradients.old, weights, nrow.weights, ncol.weights, 
              learningrate, learningrate.factor, learningrate.limit, exclude) 
    {
        weights <- unlist(weights)
        sign.gradient <- sign(gradients)
        temp <- gradients.old * sign.gradient
        positive <- temp > 0
        negative <- temp < 0
        not.negative <- !negative
        if (any(positive)) {
            learningrate[positive] <- pmin.int(learningrate[positive] * 
                                                   learningrate.factor$plus, learningrate.limit$max)
        }
        if (any(negative)) {
            weights[-exclude][negative] <- weights[-exclude][negative] + 
                gradients.old[negative] * learningrate[negative]
            learningrate[negative] <- pmax.int(learningrate[negative] * 
                                                   learningrate.factor$minus, learningrate.limit$min)
            gradients.old[negative] <- 0
            if (any(not.negative)) {
                weights[-exclude][not.negative] <- weights[-exclude][not.negative] - 
                    sign.gradient[not.negative] * learningrate[not.negative]
                gradients.old[not.negative] <- sign.gradient[not.negative]
            }
        }
        else {
            weights[-exclude] <- weights[-exclude] - sign.gradient * 
                learningrate
            gradients.old <- sign.gradient
        }
        list(gradients.old = gradients.old, weights = relist(weights, 
                                                             nrow.weights, ncol.weights), learningrate = learningrate)
    }
