# Verify Neural Nets Inputs
verify.variables <- 
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
              act.fct,
              dropout,
              visible_dropout,
              hidden_dropout) 
{
        # check if data present
        assert_is_not_null(data)
        
        # check if formula present
        assert_is_not_null(formula)
        
        # check to make sure no NA
        if(is.big.matrix(data)){
            assert_all_are_not_na(data[,])
        }else{
            assert_all_are_not_na(data)
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
            assert_is_of_length(learningrate.limit, 2)
            learningrate.limit <- as.list(learningrate.limit)
            names(learningrate.limit) <- c("min", "max")
            learningrate.limit$min <- as.vector(as.numeric(learningrate.limit$min))
            learningrate.limit$max <- as.vector(as.numeric(learningrate.limit$max))
            assert_all_are_not_na(learningrate.limit)
        }
        if (!is.null(learningrate.factor)) {
            assert_is_of_length(learningrate.factor, 2)
            learningrate.factor <- as.list(learningrate.factor)
            names(learningrate.factor) <- c("minus", "plus")
            learningrate.factor$minus <- as.vector(as.numeric(learningrate.factor$minus))
            learningrate.factor$plus <- as.vector(as.numeric(learningrate.factor$plus))
            assert_all_are_not_na(learningrate.factor)
        }else{
            learningrate.factor <- list(minus = c(0.5), plus = c(1.2))
        }
        
        # Check lifesign
        if (is.null(lifesign)) {
            lifesign <- "none"
        }
        lifesign <- as.character(lifesign)
        if (!((lifesign == "none") || (lifesign == "minimal") || 
                  (lifesign == "full"))){
            lifesign <- "minimal"
        } 
        assert_all_are_not_na(lifesign)
        
        # Check algorithm
        if (is.null(algorithm)) {
            algorithm <- "rprop+"
        }
        algorithm <- as.character(algorithm)
        if (!((algorithm == "rprop+") || (algorithm == "rprop-") || 
                  (algorithm == "slr") || (algorithm == "sag") || (algorithm == 
                                                                       "backprop"))){
            stop("'algorithm' is not known", call. = FALSE)
        } 
        if (algorithm == "backprop") {
            assert_is_numeric(learningrate.bp)
        }
        
        # Check threshold
        if (is.null(threshold)) {
            threshold <- 0.01
        }
        threshold <- as.numeric(threshold)
        assert_all_are_not_na(threshold)


        if (is.null(lifesign.step)) {
            lifesign.step <- 1000
        }
        lifesign.step <- as.integer(lifesign.step)
        assert_is_integer(lifesign.step)
        if (lifesign.step < 1){
            lifesign.step <- as.integer(100)
        } 
        
        # Check hidden
        if (is.null(hidden)) {
            hidden <- 0
        }
        hidden <- as.vector(as.integer(hidden))
        
        assert_all_are_not_na(hidden)
        assert_all_are_greater_than(hidden, 0)
        
        # Check rep
        if (is.null(rep)) {
            rep <- 1
        }
        rep <- as.integer(rep)
        assert_is_integer(rep)
        
        if (is.null(stepmax)) {
            stepmax <- 10000
        }
        stepmax <- as.integer(stepmax)
        
        if (stepmax < 1) {
            stepmax <- as.integer(1000)
        }
        
        assert_is_integer(stepmax)
        
        # Check learningrates
        if (is.null(hidden)) {
            if (is.null(learningrate.limit)) 
                learningrate.limit <- list(min = c(1e-08), max = c(50))
        }
        else {
            if (is.null(learningrate.limit)) 
                learningrate.limit <- list(min = c(1e-10), max = c(0.1))
        }
        
        # Check activation function
        if (!is.function(act.fct) && !act.fct %in% act_fcts()$act_fcts){
            stop("''act.fct' is not known", call. = FALSE)
        }
            
        # Check error function
        if (!is.function(err.fct) && !err.fct %in% c("sse", "ce")) {
            stop("'err.fct' is not known", call. = FALSE)
        }

        # Check dropout
        if(dropout){
            assert_is_of_length(hidden_dropout, length(hidden))
            assert_is_scalar(visible_dropout)
        }
        
        return(list(data = data, formula = formula, startweights = startweights, 
                    learningrate.limit = learningrate.limit, learningrate.factor = learningrate.factor, 
                    learningrate.bp = learningrate.bp, lifesign = lifesign, 
                    algorithm = algorithm, threshold = threshold, lifesign.step = lifesign.step, 
                    hidden = hidden, rep = rep, stepmax = stepmax, model.list = model.list))
    }