


training <- function(
    formula, 
    data,
    grid,
    method,
    model_args,
    verbose
    )
{
    print("called training")
    print(method)
    
    print("checking formula")
    ## Factor the class labels
    dvs <- as.character(formula[[2]])
    print("formula dv")
    print(dvs)
    #         print(colnames(data))
    dvs_idx <- which(colnames(data) %in% dvs)
    
    print("got var indicies")
    
#     data[,dvs_idx] <- factor(as.character(data[,dvs_idx]))
    xNames <- attr(terms(formula), "term.labels")
#     print("xNames:")
#     print(xNames)
    
    if(!method %in% c("neuralnet", "rf")){        
        print("ivs names acquired")
        
        trainX <- as.matrix(data[,!(names(data) %in% dvs), 
                                 drop = FALSE])
        
        mode(trainX) <- 'numeric'
        trainY <- as.factor(data[,dvs_idx])
    }
    
    if(length(unique(data[,1])) < 2){
        stop("internal dv not binary")
    }
    
#     print(table(data[,1]))
#     print(str(data[,1]))
#     stop("not training right now")
    
    print("accessing switch")
    
    mod <- switch(method,
           neuralnet = 
               {
                   if(verbose){
                       lifesign = 'full'
                   }else{
                       lifesign = 'none'
                   }
                   
                   HGmiscTools::neuralnet(formula, 
                                          data=data, 
                                          hidden = grid$.hidden, 
                                          threshold=grid$.threshold, 
                                          act.fct = "logistic", 
                                          err.fct = "ce",
                                          stepmax = 100000, 
                                          lifesign=lifesign,
                                          lifesign.step = 100,
                                          linear.output=FALSE,
                                          low_size = TRUE
                   )
           },
           gbm =  
            {
                # need to make sure only extract arguments that pertain to gbm
                gbm.args <- c("w", "var.monotone", "n.minobsinnode", 
                              "bag.fraction", "var.names", "response.name",
                              "group","n.trees","interaction.depth", 
                              "shrinkage") 
                model_args <- model_args[names(model_args) %in% gbm.args]
                
                if("n.trees" %in% names(model_args)){
                    tuneValue$.n.trees <- model_args$n.trees
                }
                if("interaciton.depth" %in% names(model_args)){
                    tuneValue$.interaction.depth <- model_args$interaction.depth
                }
                if("shrinkage" %in% names(model_args)){
                    tuneValue$.shrinkage <- model_args$shrinkage
                }
                
                if(ncol(trainX) < 50 | nrow(trainX) < 50){
                    if(is.null(model_args) | length(model_args) == 0){
                        if(nrow(trainX) < 30){
                            model_args <- list(n.minobsinnode = 2)
                        }else{
                            model_args <- list(n.minobsinnode = 5)  
                        }
                    }
                }
                
                # determine if binary or multiclass
                gbmdist <- if(length(unique(trainY)) == 2){
                    "bernoulli"}else{
                        "multinomial"
                    }         
                
                # check gbm setup file to see if this is necessary
                modY <- if(gbmdist != "multinomial") numClasses else trainY
                
                if(gbmdist != "multinomial"){
                    modY <- numClasses
                }else{
                    modY <- trainY
                }
                
                modArgs <- 
                    list(x = trainX,
                         y = modY,
                         interaction.depth = as.numeric(
                             tuneValue$.interaction.depth),
                         n.trees = as.numeric(tuneValue$.n.trees),
                         shrinkage = as.numeric(tuneValue$.shrinkage), 
                         distribution = gbmdist,
                         verbose = FALSE,
                         keep.data = FALSE)
                
                
                if(length(model_args) > 0){
                    model_args <- model_args[!names(model_args) 
                                             %in% c("n.trees", 
                                                    "interaction.depth", 
                                                    "shrinkage")]
                    modArgs <- c(modArgs, model_args)
                } 
                
                do.call("gbm.fit", modArgs)
                
            },
            
            rf =
            {             
                print("accessed rf point")
                rf.args <- c("maxnodes", "keep.forest", "keep.inbag")
                
                print("searching model args")
                print(model_args)
                
                model_args <- model_args[names(model_args) %in% rf.args]
                
                print("passed args checks")
                
#                 modArgs <- 
#                     list(x = trainX,
#                          y = trainY,
#                          importance = TRUE,
#                          mtry = as.numeric(grid$.mtry),
#                          ntree=round.multiple(sqrt(ncol(trainX)), target = 50)
#                     )
                

#               ntree=round.multiple(sqrt(length(xNames)), target = 50)
    
                modArgs <- 
                    list(formula,
                         data = data,
                         importance = TRUE,
                         mtry = as.numeric(grid$.mtry),
                         ntree=500
                    )
                
                if(length(model_args) > 0) modArgs <- c(modArgs, model_args)
                
                do.call("randomForest", modArgs)
            },
            
            svm =
            { 
                out <- svm(trainX,
                           trainY,
                           cost = as.numeric(grid[param,]$.C), 
                           cachesize=500,
                           type="C-classification", 
                           kernel="linear")                         
                out
            },
    )
    
    return(list(modelFit = mod, ivs = xNames))
}


# Round to nearest multiple of target number
round.multiple <- function(x, target, f = round) {
    out <- f(x / target) * target
    if(out == 0){
        out <- 50
    }
    out
}


