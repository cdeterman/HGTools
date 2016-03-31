

#' @export
train <-
    function(X, Y, 
             data,
             testData,
             method,
             k = 10,
             metric = "AUC",
             grid, cvControl = NULL, 
             save_models = FALSE,
             allowParallel = FALSE){
        
        assert_all_are_non_missing_nor_empty_character(X)
        assert_all_are_non_missing_nor_empty_character(Y)
        assert_is_logical(allowParallel)
        assert_all_are_positive(k)
        assert_is_data.frame(grid)
        
        if(is.null(cvControl)) cvControl = cvControl()
        nr <- nrow(data)
        
        # may only need for neuralnet, perhaps hide under some validation function
        if(is.null(cvControl$model_type)){
            if(length(attr(terms(formula.reverse), "term.labels")) == 1){
                cvControl$model_type = "binary"
            }else{
                stop("You must specify model_type!")
            }
        }
        
        if(method != "neuralnet" & !is.factor(data[,Y])){
            data[,Y] <- factor(data[,Y])
        }
        
        levs <- levels(data[,Y])
        
        formula <- as.formula(paste(paste(Y,collapse="+"), paste(X, collapse= "+"), sep="~"))
#         print(formula)
        
        # get observed values/classes
        formula.reverse <- formula
        formula.reverse[[3]] <- formula[[2]]
        
#         print(attr(terms(formula.reverse), "term.labels"))
        obs <- testData[,attr(terms(formula.reverse), "term.labels")]
        
        inTrain <- createFolds(data[,1], k = k, list = TRUE, returnTrain = TRUE)
        #print(str(inTrain))        

        outTrain <- lapply(inTrain,
                           function(inTrain, total) total[-unique(inTrain)],
                           total = seq(nr))  

        perfMatrix <- internal_cv(
            formula, data, k,
            method = method,
            grid = grid, 
            model_args = cvControl$model_args,
            model_type = cvControl$model_type,
            inTrain = inTrain, 
            outTrain = outTrain, 
            save_models = save_models,
            verbose = cvControl$verbose,
            allowParallel,
            scale = cvControl$scale) 
        
        print("internal cv complete")
        
        # sort performance matrix
        perfMatrix$means <- byComplexity(perfMatrix$means, method)
        perfMatrix$variances <- byComplexity(perfMatrix$variances, method)
        
        print(perfMatrix)
        
        # get the winner
        bestMod <- perfMatrix$means[bestFit(perfMatrix$means, metric, TRUE),]
        
        args_idx <- which(colnames(bestMod) == "AUC")
        finalGrid <- as.data.frame(bestMod[,1:(args_idx - 1)])
        colnames(finalGrid) <- names(grid)
#         colnames(finalGrid) <- gsub("\\.", "", names(grid))
        
        print(head(finalGrid))
        
        # fit the 'best' model on full dataset
        mod <- tryCatch({
            training(formula,
                     data = data, 
                     grid = finalGrid, 
                     method = method,
                     model_args = cvControl$model_args,
                     verbose = cvControl$verbose
                )
        },
        error = function(err){
            print(paste0("Error: ", err))
            stop("Full model fit failed")
        })
        

        # subset test data with only needed columns
        testDataIVs <- testData[, mod$ivs]

        print("model type?")
        print(cvControl$model_type)
        
        result <- tryCatch({
            predicting(mod$modelFit, method = method, 
                       newdata = testDataIVs, model_type = cvControl$model_type, cvControl$scale)
        },
        error = function(err){
            print(paste0("Error: ", err))
            stop("Final Internal 'predict' function failed!")
        })
        
        # some means of evaluating the model
        
        # possibly use scale01 for results???

        print(class(result))
        
#         if(method == "neuralnet"){
#             pred <- switch(class(result),
#                            nn_binary = ifelse(c(round(result@net.result)), 1, 0),
#                            stop("Only binary currently implemented")
#             )
#         }
        
#         if(method == "neuralnet"){
#             finalPerf <- predictionStats(obs, pred)
#         }else{
#             finalPerf <- predictionStats(obs, result)
#         }

        finalPerf <- predictionStats(obs, result)
        
#         print(names(mod))

#         if(save_models){
#             save(mod$modelFit, file = paste(method, "winning_model.RData", sep = "_"))
#         }
        
        out <- list(
            finalModel = mod$modelFit,
            performance = finalPerf,
            cvPerformanceMatrix = perfMatrix,
            bestParams = finalGrid
        )
        
        return(out)

#         return(perfMatrix)
    }


internal_cv <- 
    function(formula, data,
             k,
             method,
             grid, 
             model_args,
             model_type = NULL,
             inTrain, outTrain, 
             save_models,
             verbose,
             allowParallel,
             scale){
    
    `%op%` <- if(allowParallel){
        `%dopar%`
    }else{
        `%do%`
    }
    
    print("check verbose")
    print(verbose)
    
    # get observed values/classes
    formula.reverse <- formula
    formula.reverse[[3]] <- formula[[2]]
    
    obs <- data[,attr(terms(formula.reverse), "term.labels")]
    
    levs <- levels(as.factor(obs))
    
    # F1-score = 2 * (PPV * Sensitivity)/(PPV + Sensitivity)
    # AUC, Sensitivity, Specificity, PPV, NPV, F1-Score
    
    #     print("about to start foreach loop")
    #     print(seq(along = inTrain))
    #     print(seq(nrow(grid)))
    
#     print(head(data))
#     print(head(outTrain[[1]]))
#     stop()
    
    finalMetrics <- 
        foreach(iter = seq(along = inTrain)) %:%
        foreach(param = seq(nrow(grid)), .combine='rbind') %op%
    {    
        print(iter)
        print(grid[param,])
        print(iter * param)
        print(method)
#         if(iter  > 1 | param > 1){
#             stop("stopping for now")            
#         }
        
        mod <- tryCatch({
            training(formula, data[inTrain[[iter]],, drop = FALSE], 
                     grid[param,, drop = FALSE], method, 
                     model_args, 
                     verbose = verbose)
        },
        error = function(err){
            print(paste0("Error: ", err))
            stop("Internal model fit failed!")
        })
            

        if(save_models){
            save(mod, file = paste(method, paste(gsub("\\.", "", colnames(grid)),
                                                 grid[param,], sep="_", collapse="_"), 
                                   "iter", iter,
                                   "cv_model.rda", sep = "_"))
        }
        
        # subset test data with only needed columns
        testData <- data[outTrain[[iter]], mod$ivs]
        
#             print(dim(testData))
#             print(colnames(testData))
        
        result <- tryCatch({
            predicting(mod$modelFit, method, newdata = testData, model_type = model_type, model_args, param, scale)
            #HGmiscTools::compute(mod, testData, model_type=model_type),            
            #silent = TRUE
        },
        error = function(err){
            print(paste0("Error: ", err))
            stop("Internal model prediction failed!")
        })
        
        # some means of evaluating the model
        
        print("evaluate fold results")
#         print(head(obs[outTrain[[iter]]]))
#         print(head(factor(result, levels = levs)))
        
        print(predictionStats(obs[outTrain[[iter]]], result))
        print("grid param")
        print(grid[param,])

        out <- c(grid[param,], predictionStats(obs[outTrain[[iter]]], result))

        print(out)
        
        out
    } 
    
    cnames <- c(names(grid),
                "AUC", "Sensitivity", "Specificity",
                "PPV", "NPV", "F1-Score")        

    print(finalMetrics)
    
    # Take average performance across folds
    tmp <- lapply(finalMetrics, FUN=function(x) matrix(unlist(x[,(ncol(grid)+1):ncol(x)]), nrow=nrow(grid)))
#     print(tmp)
    tmp_sums <- Reduce("+", tmp)
    tmp_means <- tmp_sums/k
    means <- as.data.frame(
        matrix(
            unlist(
                cbind(finalMetrics[[1]][,1:ncol(grid)], tmp_means, deparse.level = 0)
            )
            , nrow=nrow(grid)
        )
    )
    
    colnames(means) <- cnames
    
    # Get performance variances
    tmp_diffs <- lapply(tmp, function(x) (x - tmp_means)^2)
    tmp_diff_sums <- Reduce("+", tmp_diffs)
    tmp_vars <- tmp_diff_sums/(k-1)
    vars <- as.data.frame(
        matrix(
            unlist(
                cbind(finalMetrics[[1]][,1:ncol(grid)], tmp_vars, deparse.level = 0)
            )
            , nrow=nrow(grid)
        )
    )
    colnames(vars) <- cnames
    
    out <- list(means = means, variances = vars)
    return(out)

}


byComplexity <- function(x, method)
{
    assert_is_data.frame(x)
    
    switch(method,
           neuralnet = return(x[order(x[".hidden"], -x[".threshold"]),]),
           rf = return(x[order(x[".mtry"]), ]),
           stop("unimplemented method")
           )
}


bestFit <- function(x, metric, maximize) 
{
    bestIter <- if (maximize){
        which.max(x[, metric])
    }else{
        which.min(x[, metric])
    } 
    bestIter
}

# duplication of caret::createFolds to avoid lme4 conflict
createFolds <- function(y, k = 10, list = TRUE, returnTrain = FALSE) 
{
    if (is.numeric(y)) {
        cuts <- floor(length(y)/k)
        if (cuts < 2) 
            cuts <- 2
        if (cuts > 5) 
            cuts <- 5
        breaks <- unique(quantile(y, probs = seq(0, 1, length = cuts)))
        y <- cut(y, breaks, include.lowest = TRUE)
    }
    if (k < length(y)) {
        y <- factor(as.character(y))
        numInClass <- table(y)
        foldVector <- vector(mode = "integer", length(y))
        for (i in 1:length(numInClass)) {
            min_reps <- numInClass[i]%/%k
            if (min_reps > 0) {
                spares <- numInClass[i]%%k
                seqVector <- rep(1:k, min_reps)
                if (spares > 0) 
                    seqVector <- c(seqVector, sample(1:k, spares))
                foldVector[which(y == names(numInClass)[i])] <- sample(seqVector)
            }
            else {
                foldVector[which(y == names(numInClass)[i])] <- sample(1:k, 
                                                                       size = numInClass[i])
            }
        }
    }
    else foldVector <- seq(along = y)
    if (list) {
        out <- split(seq(along = y), foldVector)
        names(out) <- paste("Fold", gsub(" ", "0", format(seq(along = out))), 
                            sep = "")
        if (returnTrain) 
            out <- lapply(out, function(data, y) y[-data], y = seq(along = y))
    }
    else out <- foldVector
    out
}
