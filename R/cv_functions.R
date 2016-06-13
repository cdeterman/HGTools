

#' @export
train <-
    function(formula, 
             data,
             testData,
             method,
             k = 10,
             metric = "AUC",
             grid, cvControl = NULL, 
             save_models = FALSE,
             allowParallel = FALSE){
        
        assert_is_formula(formula)
        assert_is_logical(allowParallel)
        assert_all_are_positive(k)
        assert_is_data.frame(grid)
        
        if(method != "neuralnet" && (is.big.matrix(data) || is.big.matrix(testData))){
            stop("only neuralnet method supports big.matrix classes")
        }
        
        if(method == "neuralnet"){
            
            if(".act_fcts" %in% colnames(grid)){
                if(!is.character(grid$.act_fcts)){
                    grid$.act_fcts <- as.character(grid$.act_fcts)
                }
            }
            
            if((".dropout" %in% colnames(grid))){
                if(any(!c(".visible_dropout", ".hidden_dropout") %in% colnames(grid))){
                    stop("grid states to use dropout but visible and/or hidden dropout parameter is missing")
                }
                
                # subset to iters with Dropout
                dropout_iters <- grid[grid$dropout == TRUE,]
                if(nrow(dropout_iters) > 0){
                    # subset to those with multiple layers
                    multiLayerCounts <- nchar(grid$.hidden) > 1
                    multiLayerIters <- dropout_iters[multiLayerCounts,]
                    
                    # make sure length of hidden_dropout matches hidden layers
                    if(nrow(multiLayerIters) > 0){
                        numDropoutLayers <- sapply(multiLayerIters$.hidden_dropout, function(x){
                            length(unlist(strsplit(x, split=",")))
                        })
                        numLayers <- sapply(multiLayerIters$.hidden, function(x){
                            length(unlist(strsplit(x, split=",")))
                        })
                        if(any(numDropoutLayers != numLayers)){
                            stop("Length of .hidden_dropout must match .hidden")
                        }
                    }
                }
                
            } 
        }
        
        if(is.null(cvControl)) cvControl = cvControl()
        nr <- nrow(data)
        
        # get ivs & dvs
        X <- attr(terms(formula), "term.labels")
        formula.reverse <- formula
        formula.reverse[[3]] <- formula[[2]]
        Y <- attr(terms(formula.reverse), "term.labels")
        
        # may only need for neuralnet, perhaps hide under some validation function
        if(is.null(cvControl$model_type)){
            #if(length(attr(terms(formula.reverse), "term.labels")) == 1){
            if(length(Y) == 1){
                cvControl$model_type = "binary"
            }else{
                cvControl$model_type = "multi"
                # stop("You must specify model_type!")
            }
        }
        
        # if not neuralnet the DV should be factor
        if(method != "neuralnet" & !is.factor(data[,Y])){
            # this is okay because can't get this far with big.matrix objects
            data[,Y] <- factor(data[,Y])
        }
        
        # get observed values/classes
        # doesn't really matter if big.matrix because likely only one column
        obs <- testData[,Y]
        
        # again, big.matrix not important as just getting indices
        inTrain <- createFolds(data[,1], k = k, list = TRUE, returnTrain = TRUE)

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
        
        # print("internal cv complete")

        for(g in seq(ncol(grid))){
            perfMatrix$means[,g] <- as(perfMatrix$means[,g], class(grid[,g]))
            perfMatrix$variances[,g] <- as(perfMatrix$variances[,g], class(grid[,g]))
        }
        
        # sort performance matrix
        perfMatrix$means <- byComplexity(perfMatrix$means, method)
        perfMatrix$variances <- byComplexity(perfMatrix$variances, method)
        
        #print(perfMatrix)
        
        # get the winner
        bestMod <- perfMatrix$means[bestFit(perfMatrix$means, metric, TRUE),]
        
        #finalGrid <- grid[0,]
        #print("empty finalGrid to fill")
        #print(str(finalGrid))

        args_idx <- which(colnames(bestMod) == "AUC")

        finalGrid <- bestMod[,1:(args_idx - 1)]

        #print("source grid str")
        #print(str(grid))

#         for(g in seq(ncol(grid))){
#             finalGrid[,g] <- as(finalGrid[,g], class(grid[,g]))
#         }
        colnames(finalGrid) <- names(grid)
        
#         # convert all to character
        #print("the finalGrid")
        #print(str(finalGrid))
#         finalGrid <- lapply(finalGrid, as.character)
        
        # fit the 'best' model on full dataset
        set.seed(as.numeric(Sys.Date()))
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
            print(err)
            stop("Full model fit failed")
        })
        

        # subset test data with only needed columns
        if(is.big.matrix(testData)){
            # reorder columns if not in order
            col_order <- match(c(mod$ivs, mod$dvs), colnames(testData))
            #orig_col_order <- as.numeric(match(colnames(testData), c(mod$ivs, mod$dvs)))
            # cols <- !colnames(testData) %in% mod$dvs
            end_col_idx <- length(mod$ivs)
            # colOrder <- as.numeric(c(col_order, dv_idx))
            if(length(unique(diff(col_order))) != 1){
                mpermuteCols(testData, order = colOrder)
            }

            testDataIVs <- sub.big.matrix(testData, firstCol = 1, lastCol = end_col_idx)
            # testDataIVs <- deepcopy(testData, cols = cols)
            # testDataIVs <- testData[][, mod$ivs]
        }else{
            testDataIVs <- testData[, mod$ivs]
        }
        
        result <- tryCatch({
            predicting(mod$modelFit, method = method, 
                       newdata = testDataIVs, model_type = cvControl$model_type, cvControl$scale)
        },
        error = function(err){
            print(paste0(err))
            stop("Final Internal 'predict' function failed!")
        })
        
        # restore original column order
        if(is.big.matrix(testData)){
            # restore original column order if it was reordered
            if(length(unique(diff(col_order))) != 1){
                mpermuteCols(testData, order = as.numeric(order(col_order)))
            }
        }
        
        # some means of evaluating the model
        finalPerf <- predictionStats(obs, as.matrix(result))
        
        out <- list(
            finalModel = mod$modelFit,
            performance = finalPerf,
            cvPerformanceMatrix = perfMatrix,
            bestParams = finalGrid
        )
        
        return(out)
    }


#' @import foreach
#' @import bigmemory
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
    
    # print("check verbose")
    # print(verbose)
    
    # get observed values/classes
    formula.reverse <- formula
    formula.reverse[[3]] <- formula[[2]]
    
    # This doesn't need to be big.matrix as likely just 1 column
    obs <- data[,attr(terms(formula.reverse), "term.labels")]
    
    levs <- levels(as.factor(obs))
    
    # F1-score = 2 * (PPV * Sensitivity)/(PPV + Sensitivity)
    # AUC, Sensitivity, Specificity, PPV, NPV, F1-Score
    
    finalMetrics <- 
        foreach(iter = seq(along = inTrain)) %:%
        foreach(param = seq(nrow(grid)), .combine='rbind') %op%
    {    
        # print(iter)
        # print(grid[param,])
        # print(iter * param)
        # print(method)
        # if(iter  > 1 | param > 1){
        #     stop("stopping for now")
        # }
        
        # start <- data[]
        if(is.big.matrix(data)){
            thread_data <- deepcopy(data)
        }
        
        
        # print(head(data))
        
        if(is.big.matrix(data)){
            all_idx <- seq(nrow(thread_data))
            sub_idx <- inTrain[[iter]]
            non_idx <- all_idx[!all_idx %in% sub_idx]
            end_idx <- length(sub_idx)
            rowOrder <- as.numeric(c(sub_idx, non_idx))
            if(length(unique(diff(rowOrder))) != 1){
                mpermute(thread_data, order = rowOrder)
            }
            subdata <- sub.big.matrix(thread_data, firstRow = 1, lastRow = end_idx)
            # subdata <- deepcopy(data, rows = inTrain[[iter]])
            # subdata <- data[][inTrain[[iter]],, drop = FALSE]
        }else{
            subdata <- data[inTrain[[iter]],, drop = FALSE]  
        }
        
        # print("training subdata")
        # print(head(subdata))
        
        # subdata2 <- data[][inTrain[[iter]],, drop = FALSE]
        # 
        # if(!all.equal(subdata, subdata2)){
        #     stop("subdata fucked up")
        # }
        
        # NA must match number of metrics returned normally
        # if(iter == 3 && param == 2){
        #     return(c(grid[param,], rep(NA, 6)))
        # }
        
        # print('training begun')
        set.seed(as.numeric(Sys.Date()))
        mod <- tryCatch({
            training(formula, subdata, 
                     grid[param,, drop = FALSE], method, 
                     model_args, 
                     verbose = verbose)
        },
        error = function(err){
            # print(err)
            warning(paste0("Internal model fit failed: \n", err))
            return(c(grid[param,], rep(NA, 6)))
            # stop("Internal model fit failed!")
        })
        
        # set.seed(42)
        # mod2 <- tryCatch({
        #     training(formula, subdata2,
        #              grid[param,, drop = FALSE], method,
        #              model_args,
        #              verbose = verbose)
        # },
        # error = function(err){
        #     print(err)
        #     stop("Internal model fit failed!")
        # })
        # 
        # if(!all.equal(mod$modelFit$weights[[1]][[2]], mod2$modelFit$weights[[1]][[2]])){
        #     stop("weights are fucked up")
        # }
        
        # print('passed initial training')
        
        if(is.big.matrix(data)){
            # restore original order
            if(length(unique(diff(rowOrder))) != 1){
                mpermute(thread_data, order = as.numeric(order(rowOrder)))
            }

            # assert_are_identical(thread_data[], start, allow_attributes = FALSE)
            # print("restored data")
            # print(head(thread_data))
        }
        

        if(save_models){
            save(mod, file = paste(method, paste(gsub("\\.", "", colnames(grid)),
                                                 grid[param,], sep="_", collapse="_"), 
                                   "iter", iter,
                                   "cv_model.rda", sep = "_"))
        }
        
        # print('passed save section')
        
        # subset test data with only needed columns
        if(is.big.matrix(data)){
            all_idx <- seq(nrow(thread_data))
            sub_idx <- outTrain[[iter]]
            non_idx <- all_idx[!all_idx %in% sub_idx]
            end_idx <- length(sub_idx)
            rowOrder <- as.numeric(c(sub_idx, non_idx))
            if(length(unique(diff(rowOrder))) != 1){
                mpermute(thread_data, order = rowOrder)
            }
            
            col_order <- match(c(mod$ivs, mod$dvs), colnames(thread_data))
            end_col_idx <- length(mod$ivs)
            # colOrder <- as.numeric(c(col_order, dv_idx))
            if(length(unique(diff(col_order))) != 1){
                mpermuteCols(thread_data, order = colOrder)
            }
            
            subdata <- sub.big.matrix(thread_data, 
                                      firstRow = 1, lastRow = length(sub_idx),
                                      firstCol = 1, lastCol = end_col_idx)
            
            # cols <- which(!colnames(data) %in% mod$dvs)
            # subdata <- deepcopy(data, rows = outTrain[[iter]], cols = cols)
            # subdata <- data[][outTrain[[iter]], mod$ivs, drop = FALSE]
        }else{
            subdata <- data[outTrain[[iter]], mod$ivs, drop = FALSE]  
        }
        
        # test_bm(subdata@address, subdata2)
        
        # subdata2 <- data[][outTrain[[iter]], mod$ivs, drop = FALSE]  
        # 
        # if(!all.equal(subdata[], subdata2)){
        #     stop("subdata fucked up")
        # }
        # stop("stop for test")
        
        # print('outTrain subdata')
        # print(head(subdata))
        #testData <- data[outTrain[[iter]], mod$ivs]
        
#             print(dim(testData))
#             print(colnames(testData))
        
        # print('about to predict')
        result <- tryCatch({
            predicting(mod$modelFit, method, newdata = subdata, 
                       model_type = model_type, model_args, param, scale)
        },
        error = function(err){
            # print(err)
            # stop("Internal model prediction failed!")
            warning(paste0("Internal model prediction failed: \n", err))
            return(c(grid[param,], rep(NA, 6)))
        })
        
        # result2 <- tryCatch({
        #     predicting(mod2$modelFit, method, newdata = subdata2, 
        #                model_type = model_type, model_args, param, scale)
        # },
        # error = function(err){
        #     print(err)
        #     stop("Internal model prediction failed!")
        # })
        # 
        # if(!all.equal(result, result2)){
        #     stop("subdata fucked up")
        # }

        # print('passed predicting')
        
        # restore original column and row order
        if(is.big.matrix(data)){
            # restore original row order
            if(length(unique(diff(rowOrder))) != 1){
                # print('restoring rows')
                mpermute(thread_data, order = as.numeric(order(rowOrder)))
            }
            # restore original column order
            if(length(unique(diff(col_order))) != 1){
                # print('restoring cols')
                mpermuteCols(thread_data, order = as.numeric(order(col_order)))
            }
            #assert_are_identical(thread_data[], start, allow_attributes = FALSE)
        }
        
        # print("restored data 2")
        # print(head(data))
        
        # some means of evaluating the model
        
#         print("evaluate fold results")
#         print(head(obs[outTrain[[iter]]]))
#         print(head(factor(result, levels = levs)))
        
#         print(predictionStats(obs[outTrain[[iter]]], result))
#         print("grid param")
#         print(grid[param,])

        out <- c(grid[param,], predictionStats(obs[outTrain[[iter]]], result))

#         print(out)
        
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
