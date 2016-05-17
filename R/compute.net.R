# possible area of improvment
compute.net <-
    function (weights, 
              length.weights, 
              covariate, # big.matrix
              act.fct, act.deriv.fct, 
              output.act.fct, output.act.deriv.fct, special) 
    {
        neuron.deriv <- NULL
        neurons <- list(covariate)
        if (length.weights > 1) {
            for (i in 1:(length.weights - 1)) {
                
                # Other major bottleneck point
                if(class(neurons[[i]]) == 'big.matrix'){
                    # temp <- neurons[[i]][,] %*% weights[[i]]  
                    temp <- neurons[[i]] %*% weights[[i]]  
                }else{
                    temp <- neurons[[i]] %*% weights[[i]]
                }
                
                
                # convert to normal matrix/numeric type because smaller
                # need to find a way of applying sign changes and 'exp' function
                # act.temp <- act.fct(temp[,])
                act.temp <- act.fct(temp)
                if (special) {
                    #print("special")
                    #neuron.deriv[[i]] <- as.matrix(act.deriv.fct(act.temp))
                    neuron.deriv[[i]] <- act.deriv.fct(act.temp)
                }else{
                    #print("not special")
                    # neuron.deriv[[i]] <- act.deriv.fct(temp[,])
                    neuron.deriv[[i]] <- act.deriv.fct(temp)
                }
                
                # big.matrix requires manual cbind
                neurons[[i + 1]] <- big.matrix(nrow = nrow(act.temp), 
                                               ncol = ncol(act.temp),
                                               init = 1)
                neurons[[i + 1]][,2:(ncol(act.temp) + 1)] <- act.temp[]
                #neurons[[i+1]] <- cbindBM(act.temp, 1, binding="left")
            }
        }
        if (!is.list(neuron.deriv)) {
            neuron.deriv <- list(neuron.deriv)
        }
        temp <- neurons[[length.weights]] %*% weights[[length.weights]]
        net.result <- output.act.fct(temp)
        if (special) {
            neuron.deriv[[length.weights]] <- output.act.deriv.fct(net.result)
        }else{
            neuron.deriv[[length.weights]] <- output.act.deriv.fct(temp)
        }
        if (any(is.na(neuron.deriv))) {
            stop("neuron derivatives contain a NA; varify that the derivative function does not divide by 0", 
                 call. = FALSE)
        }
        
        return(list(neurons = neurons, neuron.deriv = neuron.deriv, net.result = net.result))
    }

