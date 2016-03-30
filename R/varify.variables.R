# Verify Neural Nets Inputs
varify.variables <- 
    function (data, 
              formula, 
              startweights, 
              learningrate.limit, 
              learningrate.factor, 
              learningrate.bp, 
              lifesign, 
              algorithm, 
              threshold, 
              lifesign.step, 
              hidden, 
              rep, 
              stepmax, 
              err.fct, 
              act.fct) 
{
        # check if data present
        if (is.null(data)) 
            stop("'data' is missing", call. = FALSE)
        # check if formula present
        if (is.null(formula)) 
            stop("'formula' is missing", call. = FALSE)
        
        # check to make sure no NA
        if(is.big.matrix(data)){
            if(any(is.na(data[,]))){
                stop("'data' contains NA values")
            }
        }else{
            if(any(is.na(data))){
                stop("'data' contains NA values")
            }
        }
        
        
        if (!is.null(startweights)) {
            startweights <- as.vector(unlist(startweights))
            if (any(is.na(startweights))) 
                startweights <- startweights[!is.na(startweights)]
        }
        if(class(data)!="big.matrix"){
            data <- as.data.frame(data)
        }
        formula <- as.formula(formula)
        model.vars <- attr(terms(formula), "term.labels")
        formula.reverse <- formula
        formula.reverse[[3]] <- formula[[2]]
        model.resp <- attr(terms(formula.reverse), "term.labels")
        model.list <- list(response = model.resp, variables = model.vars)
        if (!is.null(learningrate.limit)) {
            if (length(learningrate.limit) != 2) 
                stop("'learningrate.factor' must consist of two components", 
                     call. = FALSE)
            learningrate.limit <- as.list(learningrate.limit)
            names(learningrate.limit) <- c("min", "max")
            learningrate.limit$min <- as.vector(as.numeric(learningrate.limit$min))
            learningrate.limit$max <- as.vector(as.numeric(learningrate.limit$max))
            if (is.na(learningrate.limit$min) || is.na(learningrate.limit$max)) 
                stop("'learningrate.limit' must be a numeric vector", 
                     call. = FALSE)
        }
        if (!is.null(learningrate.factor)) {
            if (length(learningrate.factor) != 2) 
                stop("'learningrate.factor' must consist of two components", 
                     call. = FALSE)
            learningrate.factor <- as.list(learningrate.factor)
            names(learningrate.factor) <- c("minus", "plus")
            learningrate.factor$minus <- as.vector(as.numeric(learningrate.factor$minus))
            learningrate.factor$plus <- as.vector(as.numeric(learningrate.factor$plus))
            if (is.na(learningrate.factor$minus) || is.na(learningrate.factor$plus)) 
                stop("'learningrate.factor' must be a numeric vector", 
                     call. = FALSE)
        }
        else learningrate.factor <- list(minus = c(0.5), plus = c(1.2))
        if (is.null(lifesign)) 
            lifesign <- "none"
        lifesign <- as.character(lifesign)
        if (!((lifesign == "none") || (lifesign == "minimal") || 
                  (lifesign == "full"))) 
            lifesign <- "minimal"
        if (is.na(lifesign)) 
            stop("'lifesign' must be a character", call. = FALSE)
        if (is.null(algorithm)) 
            algorithm <- "rprop+"
        algorithm <- as.character(algorithm)
        if (!((algorithm == "rprop+") || (algorithm == "rprop-") || 
                  (algorithm == "slr") || (algorithm == "sag") || (algorithm == 
                                                                       "backprop"))) 
            stop("'algorithm' is not known", call. = FALSE)
        if (is.null(threshold)) 
            threshold <- 0.01
        threshold <- as.numeric(threshold)
        if (is.na(threshold)) 
            stop("'threshold' must be a numeric value", call. = FALSE)
        if (algorithm == "backprop") 
            if (is.null(learningrate.bp) || !is.numeric(learningrate.bp)) 
                stop("'learningrate' must be a numeric value, if the backpropagation algorithm is used", 
                     call. = FALSE)
        if (is.null(lifesign.step)) 
            lifesign.step <- 1000
        lifesign.step <- as.integer(lifesign.step)
        if (is.na(lifesign.step)) 
            stop("'lifesign.step' must be an integer", call. = FALSE)
        if (lifesign.step < 1) 
            lifesign.step <- as.integer(100)
        if (is.null(hidden)) 
            hidden <- 0
        hidden <- as.vector(as.integer(hidden))
        if (prod(!is.na(hidden)) == 0) 
            stop("'hidden' must be an integer vector or a single integer", 
                 call. = FALSE)
        if (length(hidden) > 1 && prod(hidden) == 0) 
            stop("'hidden' contains at least one 0", call. = FALSE)
        if (is.null(rep)) 
            rep <- 1
        rep <- as.integer(rep)
        if (is.na(rep)) 
            stop("'rep' must be an integer", call. = FALSE)
        if (is.null(stepmax)) 
            stepmax <- 10000
        stepmax <- as.integer(stepmax)
        if (is.na(stepmax)) 
            stop("'stepmax' must be an integer", call. = FALSE)
        if (stepmax < 1) 
            stepmax <- as.integer(1000)
        if (is.null(hidden)) {
            if (is.null(learningrate.limit)) 
                learningrate.limit <- list(min = c(1e-08), max = c(50))
        }
        else {
            if (is.null(learningrate.limit)) 
                learningrate.limit <- list(min = c(1e-10), max = c(0.1))
        }
        if (!is.function(act.fct) && act.fct != "logistic" && act.fct != 
                "tanh") 
            stop("''act.fct' is not known", call. = FALSE)
        if (!is.function(err.fct) && err.fct != "sse" && err.fct != 
                "ce") 
            stop("'err.fct' is not known", call. = FALSE)
        return(list(data = data, formula = formula, startweights = startweights, 
                    learningrate.limit = learningrate.limit, learningrate.factor = learningrate.factor, 
                    learningrate.bp = learningrate.bp, lifesign = lifesign, 
                    algorithm = algorithm, threshold = threshold, lifesign.step = lifesign.step, 
                    hidden = hidden, rep = rep, stepmax = stepmax, model.list = model.list))
    }