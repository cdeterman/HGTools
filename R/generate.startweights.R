# no big.matrix components
generate.startweights <-
    function (model.list, hidden, startweights, rep, exclude, constant.weights) 
    {
        input.count <- length(model.list$variables)
        output.count <- length(model.list$response)
        if (!(length(hidden) == 1 && hidden == 0)) {
            length.weights <- length(hidden) + 1
            nrow.weights <- array(0, dim = c(length.weights))
            ncol.weights <- array(0, dim = c(length.weights))
            nrow.weights[1] <- (input.count + 1)
            ncol.weights[1] <- hidden[1]
            if (length(hidden) > 1) {
                for (i in 2:length(hidden)) {
                    nrow.weights[i] <- hidden[i - 1] + 1
                    ncol.weights[i] <- hidden[i]
                }
            }
            nrow.weights[length.weights] <- hidden[length.weights - 
                                                       1] + 1
            ncol.weights[length.weights] <- output.count
        }
        else {
            length.weights <- 1
            nrow.weights <- array((input.count + 1), dim = c(1))
            ncol.weights <- array(output.count, dim = c(1))
        }
        length <- sum(ncol.weights * nrow.weights)
        vector <- rep(0, length)
        if (!is.null(exclude)) {
            if (is.matrix(exclude)) {
                exclude <- matrix(as.integer(exclude), ncol = ncol(exclude), 
                                  nrow = nrow(exclude))
                if (nrow(exclude) >= length || ncol(exclude) != 3) 
                    stop("'exclude' has wrong dimensions", call. = FALSE)
                if (any(exclude < 1)) 
                    stop("'exclude' contains at least one invalid weight", 
                         call. = FALSE)
                temp <- relist(vector, nrow.weights, ncol.weights)
                for (i in 1:nrow(exclude)) {
                    if (exclude[i, 1] > length.weights || exclude[i, 
                                                                  2] > nrow.weights[exclude[i, 1]] || exclude[i, 
                                                                                                              3] > ncol.weights[exclude[i, 1]]) 
                        stop("'exclude' contains at least one invalid weight", 
                             call. = FALSE)
                    temp[[exclude[i, 1]]][exclude[i, 2], exclude[i, 
                                                                 3]] <- 1
                }
                exclude <- which(unlist(temp) == 1)
            }
            else if (is.vector(exclude)) {
                exclude <- as.integer(exclude)
                if (max(exclude) > length || min(exclude) < 1) {
                    stop("'exclude' contains at least one invalid weight", 
                         call. = FALSE)
                }
            }
            else {
                stop("'exclude' must be a vector or matrix", call. = FALSE)
            }
            if (length(exclude) >= length) 
                stop("all weights are exluded", call. = FALSE)
        }
        length <- length - length(exclude)
        if (!is.null(exclude)) {
            if (is.null(startweights) || length(startweights) < (length * 
                                                                     rep)){
                vector[-exclude] <- rnorm(length)
            }else{
                vector[-exclude] <- startweights[((rep - 1) * length + 
                                                      1):(length * rep)]
            } 
        }else{
            if (is.null(startweights) || length(startweights) < (length * 
                                                                     rep)) {
                vector <- rnorm(length) 
            }else{
                vector <- startweights[((rep - 1) * length + 1):(length * 
                                                                     rep)]
            } 
        }
        if (!is.null(exclude) && !is.null(constant.weights)) {
            if (length(exclude) < length(constant.weights)) {
                stop("constant.weights contains more weights than exclude", 
                     call. = FALSE)
            }else{
                vector[exclude[1:length(constant.weights)]] <- constant.weights
            }
        }
        weights <- relist(vector, nrow.weights, ncol.weights)
        return(list(weights = weights, exclude = exclude))
    }
