#' @title Model Group Prediction
#' @description This function evaluates a single fitted model and returns 
#' the predicted group memberships.
#' @param method String of the model to be evaluated
#' @param modelFit The fitted model being evaluated
#' @param newdata The testing data to predict group membership
#' @param model_args Additional arguments for individual models
#' @param param The parameters being fit to the model 
#' (Determined by model optimization).
#' @return Returns a list of predicted group membership
#' @import randomForest
#' @import e1071
#' @import gbm
#' @import glmnet
# ' @export

predicting <- function(modelFit, method, newdata, model_type = NULL, model_args = NULL, param = NULL, scale = FALSE)
{
#     if(any(colnames(newdata) == ".classes")) newdata$.classes <- NULL
    
    # print("trying to predict")
    # print(method)
    # print(model_type)
    # print(model_args)
    # print(dim(newdata))
    # print("compute data")
    # print(head(newdata))
    # print("modelFit weights")
    # print(modelFit$weights[[1]][[2]])
    
    predictedValue <- 
        switch(method,   
               
        neuralnet = 
        {
            # print("about to 'compute'")
            # result <- HGTools::compute(modelFit, covariate = newdata, model_type=model_type)
            result <- predict(modelFit, newdata = newdata, type = "prob")
            
            # print('compute passed')
            # possibly use scale01 for results???
            if(model_type == "binary"){
                if(scale){
                    out <- scale01(result)
                    # out <- ifelse(c(round(scale01(result@net.result))), 1, 0)
                    print("scaled results")
                }else{
                    out <- result
                }
                #         pred <- ifelse(c(round(result@net.result)), 1, 0)
            }else{
                stop("Only binary currently implemented")
            }
            # print('compute finished')
            out
        },
        
        gbm =
        {
            gbmProb <- predict(modelFit, newdata, type = "response",
                               n.trees = modelFit$tuneValue$.n.trees)
            gbmProb[is.nan(gbmProb)] <- NA
            
            # need a check if all NA
            # if so, n.trees are way too high
            
            if(modelFit$distribution$name != "multinomial")
            {
                out <- ifelse(gbmProb >= .5, 
                              modelFit$obsLevels[1], 
                              modelFit$obsLevels[2])
                ## to correspond to gbmClasses definition above
            } else {
                out <- colnames(gbmProb)[apply(gbmProb, 1, which.max)]
            }
            
            # if there is a parameter that multiple models can be drawn, 
            # extract these other 'lower' models
            if(!is.null(param))
            {
                tmp <- predict(modelFit, newdata, 
                               type = "response", n.trees = param$.n.trees)
                
                if(modelFit$distribution$name != "multinomial"){
                    # if only one other parameter, need to convert to matrix
                    if(is.vector(tmp)) tmp <- matrix(tmp, ncol = 1)
                    tmp <- apply(tmp, 2,
                                 function(x, nm = modelFit$obsLevels){
                                     ifelse(x >= .5, nm[1], nm[2])
                                 })
                }else{
                    tmp <- apply(tmp, 3,
                                 function(y, nm = modelFit$obsLevels){
                                     nm[apply(y, 1, which.max)]
                                 })
                }
                
                # convert to list compatible splits
                if(length(tmp) > 1){
                    if(!is.list(tmp)) tmp <- split(tmp, 
                                                   rep(1:ncol(tmp), 
                                                       each = nrow(tmp)))
                }
                out <- c(list(out), tmp)
            }
            out
        },
        
        rf =
        {
            print("rf predict start")
            #print(str(modelFit))
            tryCatch({
                out <-  predict(modelFit, newdata, type = "prob")[,2]
                print("rf predict finished")
                return(out)
            }, error = function(err){
                print(paste("MY_ERROR: ", err))
                stop()
            })
            
            #out
        },
        
        svm =                           
        {                          
            out <- as.character(predict(modelFit, newdata = newdata))
            out
        },
        
        glmnet =
        {  
            #     print("new data input")
            #     print(head(newdata))
            if(!is.matrix(newdata)) newdata <- as.matrix(newdata)
            
            if(!is.null(param))
            {
                #print(param)
                #         print(head(newdata))
                out <- predict(modelFit, newdata, 
                               s = param$.lambda, type = "class")
                out <- as.list(as.data.frame(out, stringsAsFactors = FALSE))
            } else {
                
                if(is.null(modelFit$lambdaOpt))
                    stop("optimal lambda not saved; 
                                needs a single lambda value")
                
                out <- predict(modelFit, newdata, 
                               s = modelFit$lambdaOpt, type = "class")[,1]
            }
            out
        },
        
        stop("unrecognized model")

        )
    
    # print(head(predictedValue))
    
    return(predictedValue)
}

