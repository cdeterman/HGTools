
#' @importFrom bigmemory typeof
#' @export
fast_neuralnet_bm <-
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
      # print("called fast bm nn")
    call <- match.call()
    options(scipen = 100, digits = 10)
    
    if(typeof(data) != 'double'){
      stop("Big Matrix objects must be of type 'double'")
    }
    
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
    out_matrix <- NULL
    list.result <- NULL
    
    # create bigMatrix for covariate and response
    covariate <- big.matrix(nrow=nrow(data), 
                               ncol=1+length(model.list[[2]]), 
                               type="double",
                               shared = FALSE,
                               init = 1)
    response <- big.matrix(nrow=nrow(data), 
                              ncol=length(model.list[[1]]), 
                              type="double",
                              shared = FALSE,
                              init = 0)
    
    # print("data")
    # print(head(data[]))
    
    covariate[,2:ncol(covariate)] <- data[,which(colnames(data) %in% model.list$variables)]
    response[] <- data[,which(colnames(data) %in% model.list$response)]
    
    # generate initial variables
    result = c_generate_initial_variables_bm(
      response@address,
      model.list, act.fct, err.fct)
    
    
    # print(head(response))
    
    # stop("stopping")
    
    
    # the response and covariate BigMatrix objects were modified 
    # within c_generate_initial_variables_bm
    #covariate <- covariate.BM
    #response <- response.BM
    err.fct <- result$err.fct               # XPtr
    err.deriv.fct <- result$err.deriv.fct   # XPtr
    act.fct <- result$act.fct               # XPtr
    act.deriv.fct <- result$act.deriv.fct   # XPtr
    
    
    for (i in 1:rep) {
      if (lifesign != "none") {
        tmp <- display(hidden, threshold, rep, i, lifesign)
      }
      flush.console()
      
      # print("just before calculate")
      
      # calculate neuralnet scores 
      result <-
        c_calculate_neuralnet_bm(
          data = data@address,       
          model_list = model.list, 
          hidden = hidden, 
          stepmax = stepmax,
          rep = i,
          threshold = threshold, 
          learningrate_limit = learningrate.limit, 
          learningrate_factor = learningrate.factor, 
          lifesign = lifesign, 
          covariate = covariate@address,  
          response = response@address,  
          lifesign_step = lifesign.step, 
          startweights = startweights,
          algorithm = algorithm, 
          act_fct = act.fct, act_deriv_fct = act.deriv.fct, act_fct_name = act.fct.name,
          err_fct = err.fct, err_deriv_fct = err.deriv.fct, err_fct_name = err.fct.name, 
          linear_output = linear.output, 
          likelihood = likelihood, 
          exclude = exclude, 
          constant_weights = constant.weights, 
          learningrate_bp = learningrate.bp,
          dropout = dropout,
          visible_dropout = visible_dropout,
          hidden_dropout = hidden_dropout)
      
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
        out_matrix <- cbind(out_matrix, result$output.vector)
      }
    }
    
    flush.console()
    if (!is.null(out_matrix)) {
      weight.count <- length(unlist(list.result[[1]]$weights)) - 
        length(exclude) + length(constant.weights) - sum(constant.weights == 
                                                           0)
      if (!is.null(startweights) && length(startweights) < 
            (rep * weight.count)) {
        warning("some weights were randomly generated, because 'startweights' did not contain enough values", 
                call. = F)
      }
      ncol.matrix <- ncol(out_matrix)
    }else{
      ncol.matrix <- 0
    }
    
    if(ncol.matrix < rep){warning(sprintf("algorithm did not converge in %s of %s repetition(s) within the stepmax", 
                                          (rep - ncol.matrix), rep), call. = FALSE)}
    
    # generate formatted output
    nn <- generate.output(covariate, call, rep, threshold, out_matrix, 
                          startweights, model.list, response, err.fct.name, act.fct.name, 
                          data, list.result, linear.output, exclude, low_size,
                          dropout, visible_dropout, hidden_dropout)
    # change class from 'nn' to 'fnn'
    class(nn) <- c('fnn')
    return(nn)
  }