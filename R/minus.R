minus <-
    function (gradients, gradients.old, weights, length.weights, 
              nrow.weights, ncol.weights, learningrate, learningrate.factor, 
              learningrate.limit, algorithm, exclude) 
    {
        weights <- unlist(weights)
        temp <- gradients.old * gradients
        positive <- temp > 0
        negative <- temp < 0
        if (any(positive)) 
            learningrate[positive] <- pmin.int(learningrate[positive] * 
                                                   learningrate.factor$plus, learningrate.limit$max)
        if (any(negative)) 
            learningrate[negative] <- pmax.int(learningrate[negative] * 
                                                   learningrate.factor$minus, learningrate.limit$min)
        if (algorithm != "rprop-") {
            delta <- 10^-6
            notzero <- gradients != 0
            gradients.notzero <- gradients[notzero]
            if (algorithm == "slr") {
                min <- which.min(learningrate[notzero])
            }
            else if (algorithm == "sag") {
                min <- which.min(abs(gradients.notzero))
            }
            if (length(min) != 0) {
                temp <- learningrate[notzero] * gradients.notzero
                sum <- sum(temp[-min]) + delta
                learningrate[notzero][min] <- min(max(-sum/gradients.notzero[min], 
                                                      learningrate.limit$min), learningrate.limit$max)
            }
        }
        weights[-exclude] <- weights[-exclude] - sign(gradients) * 
            learningrate
        list(gradients.old = gradients, weights = relist(weights, 
                                                         nrow.weights, ncol.weights), learningrate = learningrate)
    }
