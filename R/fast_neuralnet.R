#' @importFrom Rcpp evalCpp
#' @useDynLib HGTools
#' @export
fast_neuralnet <-
  function (formula, 
            data, 
            hidden = 1, 
            threshold = 0.01, 
            stepmax = 1e+05, 
            rep = 1, 
            startweights = NULL, 
            learningrate.limit = NULL, 
            learningrate.factor = list(minus = 0.5, plus = 1.2), 
            learningrate = NULL, 
            lifesign = "none", 
            lifesign.step = 1000, 
            algorithm = "rprop+", 
            err.fct = "sse", 
            act.fct = "logistic", 
            linear.output = TRUE, 
            exclude = NULL, 
            constant.weights = NULL, 
            likelihood = FALSE,
            low_size = TRUE,
            dropout = FALSE,
            visible_dropout = 0,
            hidden_dropout = rep(0, length(hidden))) 
{
      print("called fast_neuralnet")
    call <- match.call()
    options(scipen = 100, digits = 10)
    
    # verify inputs are appropriate
    result <- verify.variables(data, formula, startweights, learningrate.limit, 
                               learningrate.factor, learningrate, lifesign, algorithm, 
                               threshold, lifesign.step, hidden, rep, stepmax, err.fct, 
                               act.fct, dropout, visible_dropout, hidden_dropout)
    data <- result$data
    formula <- result$formula
    startweights <- result$startweights
    learningrate.limit <- result$learningrate.limit
    learningrate.factor <- result$learningrate.factor
    learningrate.bp <- result$learningrate.bp
    lifesign <- result$lifesign
    algorithm <- result$algorithm
    threshold <- result$threshold
    lifesign.step <- result$lifesign.step
    hidden <- result$hidden
    rep <- result$rep
    stepmax <- result$stepmax
    model.list <- result$model.list
    err.fct.name <- err.fct
    act.fct.name <- act.fct
    
    # empty objects to fill
    matrix <- NULL
    list.result <- NULL
    
    # generate initial variables
    result = c_generate_initial_variables(
      data, model.list, 
      act.fct, err.fct)
    
    covariate <- result$covariate
    response <- result$response
    err.fct <- result$err.fct               
    err.deriv.fct <- result$err.deriv.fct   
    act.fct <- result$act.fct               
    act.deriv.fct <- result$act.deriv.fct   
    
    for (i in 1:rep) {
      if (lifesign != "none") {
        tmp <- display(hidden, threshold, rep, i, lifesign)
      }
      flush.console()
      
      # calculate neuralnet scores 
      #set.seed(123)
      # print("calling calculate")
      result <-
        c_calculate_neuralnet(
          data, 
          model.list, 
          hidden, 
          stepmax,
          rep = i,
          threshold, 
          learningrate.limit, 
          learningrate.factor, 
          lifesign, 
          covariate,
          response,
          lifesign.step, 
          startweights,
          algorithm, 
          act.fct, act.deriv.fct, act.fct.name,
          err.fct, err.deriv.fct, err.fct.name, 
          linear.output, 
          likelihood, 
          exclude, 
          constant.weights, 
          learningrate.bp,
          dropout,
          visible_dropout,
          hidden_dropout)
      
      #result
      # add results to list object
      if (!is.null(result$output.vector)) {
        list.result <- c(list.result, list(result))
        if(!likelihood){
          row.names(result$output.vector) <- 
            c("error","reach_threshold","steps",
              rep("", nrow(result$output.vector) - 3))
        }else{
          row.names(result$output.vector) <- 
            c("error","reach_threshold","steps","aic","bic", 
              rep("", nrow(result$output.vector) - 5))
        }
        matrix <- cbind(matrix, result$output.vector)
      }
    }
    
    flush.console()
    if (!is.null(matrix)) {
      weight.count <- length(unlist(list.result[[1]]$weights)) - 
        length(exclude) + length(constant.weights) - sum(constant.weights == 
                                                           0)
      if (!is.null(startweights) && length(startweights) < 
            (rep * weight.count)) {
        warning("some weights were randomly generated, because 'startweights' did not contain enough values", 
                call. = F)
      }
      ncol.matrix <- ncol(matrix)
    }else{
      ncol.matrix <- 0
    }
    
    if(ncol.matrix < rep){warning(sprintf("algorithm did not converge in %s of %s repetition(s) within the stepmax", 
                                          (rep - ncol.matrix), rep), call. = FALSE)}
    
    # generate formatted output
    nn <- generate.output(covariate, call, rep, threshold, matrix, 
                          startweights, model.list, response, err.fct.name, act.fct.name, 
                          data, list.result, linear.output, exclude, low_size,
                          dropout, visible_dropout, hidden_dropout)
    # change class from 'nn' to 'fnn'
    class(nn) <- c('fnn')
    
    print("passed neuralnet")
    return(nn)
  }