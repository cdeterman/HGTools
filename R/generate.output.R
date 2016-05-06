
# generating neuralnet output objects
generate.output <-
    function (covariate, call, rep, threshold, matrix, startweights, 
              model.list, response, err.fct, act.fct, data, list.result, 
              linear.output, exclude, low_size, 
              dropout, visible_dropout, hidden_dropout) 
    {
        covariate <- remove.intercept(covariate)
        nn <- list(call = call)
        class(nn) <- c("nn")
        
        nn$model.list <- model.list
        nn$act.fct <- act.fct
        nn$linear.output <- linear.output
        
        if(!low_size){
            nn$response <- response
            nn$covariate <- covariate
            nn$err.fct <- err.fct
            nn$data <- data
            nn$exclude <- exclude
            if (!is.null(matrix)) {
                nn$net.result <- NULL
                nn$weights <- NULL
                nn$generalized.weights <- NULL
                nn$startweights <- NULL
                for (i in 1:length(list.result)) {
                    nn$net.result <- c(nn$net.result, list(list.result[[i]]$net.result))
                    nn$weights <- c(nn$weights, list(list.result[[i]]$weights))
                    nn$startweights <- c(nn$startweights, list(list.result[[i]]$startweights))
                    nn$generalized.weights <- c(nn$generalized.weights, 
                                                list(list.result[[i]]$generalized.weights))
                }
                nn$result.matrix <- generate.rownames(matrix, nn$weights[[1]], 
                                                      model.list)
            }
            nn$dropout = dropout
            nn$visible_dropout = visible_dropout
            nn$hidden_dropout = hidden_dropout
            
        }else{
            if (!is.null(matrix)) {
                nn$weights <- NULL
                for (i in 1:length(list.result)) {
                    nn$weights <- c(nn$weights, list(list.result[[i]]$weights))
                }
            }
            nn$dropout = dropout
            nn$visible_dropout = visible_dropout
            nn$hidden_dropout = hidden_dropout
        }
        
        return(nn)
    }
