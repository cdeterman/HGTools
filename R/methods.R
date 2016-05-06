
# ' @rdname neuralnet-methods
# ' @aliases neuralnet,big.matrix-method
# ' @export
setMethod("neuralnet", signature("formula", "big.matrix"), 
          function(formula, data, hidden=1, threshold=0.01, stepmax=1e+05, 
                   rep=1, startweights=NULL, learningrate.limit = NULL,
                   learningrate.factor = list(minus = 0.5, plus = 1.2), 
                   learningrate=NULL, lifesign = "none", 
                   lifesign.step = 1000, algorithm = "rprop+", 
                   err.fct = "sse", act.fct = "logistic", 
                   linear.output = TRUE, exclude = NULL, 
                   constant.weights = NULL, likelihood = FALSE,
                   low_size = TRUE){
            
            # check act.fct is function
            # This is essentially the filter between the
            # fast (Rcpp) and default (R) implementation
            # The fast function requires a string on a 
            # defined function
            if(!is.function(act.fct)){
              if(is.character(act.fct)){
                if(!act.fct %in% act_fcts()[,1]){
                  stop("Unrecognized activation function (act.fct).
                       Check options with act_fct().")
                }
                }else{
                  stop("Unrecognized activation function (act.fct)")
              }
            }
            
            if(is.function(act.fct)){
              big.neuralnet(formula, data, hidden, threshold,        
                            stepmax, rep, startweights, 
                            learningrate.limit, learningrate.factor, 
                            learningrate, lifesign, lifesign.step, 
                            algorithm, err.fct, act.fct, 
                            linear.output, exclude, 
                            constant.weights, likelihood, low_size)
            }else{
              fast_neuralnet_bm(formula, data, hidden, threshold,        
                                stepmax, rep, startweights, 
                                learningrate.limit, learningrate.factor, 
                                learningrate, lifesign, lifesign.step, 
                                algorithm, err.fct, act.fct, 
                                linear.output, exclude, 
                                constant.weights, likelihood, low_size)
            }
            
            })


# ' @rdname neuralnet-methods
# ' @aliases neuralnet,matrixORdataframe-method
# ' @export
setMethod("neuralnet", signature("formula", "matrixORdataframe"), 
          function(formula, data, hidden=1, threshold=0.01, stepmax=1e+05, 
                   rep=1, startweights=NULL, learningrate.limit = NULL,
                   learningrate.factor = list(minus = 0.5, plus = 1.2), 
                   learningrate=NULL, lifesign = "none", 
                   lifesign.step = 1000, algorithm = "rprop+", 
                   err.fct = "sse", act.fct = "logistic", 
                   linear.output = TRUE, exclude = NULL, 
                   constant.weights = NULL, likelihood = FALSE,
                   low_size = TRUE,
                   dropout = False,
                   visible_dropout = NULL,
                   hidden_dropout = NULL){
            
            # check act.fct is function
            # This is essentially the filter between the
            # fast (Rcpp) and default (R) implementation
            # The fast function requires a string on a 
            # defined function
            if(!is.function(act.fct)){
              if(is.character(act.fct)){
                if(!act.fct %in% act_fcts()[,1]){
                  stop("Unrecognized activation function (act.fct).
                       Check options with act_fct().")
                }
                }else{
                  stop("Unrecognized activation function (act.fct)")
              }
            }
            
            if(is.function(act.fct)){
                if(low_size){
                    warning("Using default neuralnet package, output may be large")
                }
                if(dropout){
                    stop("The default neuralnet package does not support dropout.
                         Please use an internal activation function (see act_fcts()).")
                }
                out <- neuralnet::neuralnet(formula, data, hidden, threshold,        
                                            stepmax, rep, startweights, 
                                            learningrate.limit, 
                                            learningrate.factor, 
                                            learningrate, lifesign, 
                                            lifesign.step, algorithm, 
                                            err.fct, act.fct, 
                                            linear.output, exclude, 
                                            constant.weights, likelihood)
            }else{
                out <- fast_neuralnet(formula, data, hidden, threshold,        
                                      stepmax, rep, startweights, 
                                      learningrate.limit, 
                                      learningrate.factor, 
                                      learningrate, lifesign, 
                                      lifesign.step, algorithm, 
                                      err.fct, act.fct, 
                                      linear.output, exclude, 
                                      constant.weights, likelihood,
                                      low_size,
                                      dropout,
                                      visible_dropout,
                                      hidden_dropout)
            }
            
            class(out) <- c("list")
            return(out)
            })

## Not currently specifying big.matrix within 'x'.  The only components
## used from neuralnet are weights, act.fct, and linear.output.  The latter
## two are a function and logical respectively.  Weights is currenty always
## returned a matrix because it is never very large.

# ' @export
setMethod("compute", signature("list", "matrixORdataframe"), 
          function(x, covariate, rep=1, model_type=NULL) {
            err_check <- check_nn_object(x)
#             if(!err_check){
#               stop(err_check)
#             }
            
            if(is.null(model_type)){
                dvs <- x$model.list[["response"]]
                if(length(dvs) > 1){
                    model_type <- "multi"
                }
                if(length(dvs) == 1){
                    model_type <- "binary"
                }
                
                if(!model_type %in% model_types()[,1]){
                    stop("Error: 'model_type' not defined")
                }
            }
            
            if(!model_type %in% model_types()[,1]){
              stop("Error: 'model_type' not defined")
            }
            
            # check act.fct is function
            if(!is.function(x[["act.fct"]])){
              if(is.character(x[["act.fct"]])){
                if(!x[["act.fct"]] %in% act_fcts()[,1]){
                  stop("Unrecognized activation function (act.fct).
                       Check options with act_fct().")
                }
                }else{
                  stop("Unrecognized activation function (act.fct)")
              }
            }
            
            if(is.function(x[["act.fct"]])){
              tmp <- neuralnet::compute(x, covariate, rep)
            }else{
              tmp <- fast_compute(x, covariate, rep)
            }
            
            out <- new(paste0("nn_", model_type), neurons=tmp$neurons, 
                       net.result=tmp$net.result,
                       dvs = if(is.null(x$model.list$response)) "unknown" else x$model.list$response,
                       ivs = if(is.null(x$model.list$variables)) colnames(covariate) else x$model.list$variables)
            return(out)
            })

# ' @export
setMethod("compute", signature("list", "big.matrix"), 
          function(x, covariate, rep=1, model_type=NULL) {
            err_check <- check_nn_object(x)
#             if(!err_check){
#               stop(err_check)
#             }
            
            if(is.null(model_type)){
                #dvs <- x$model.list[["response"]]
                #if(length(dvs) > 1){
                #    model_type <- "multi"
                #}
                #if(length(dvs) == 1){
                #    model_type <- "binary"
                #}
                warning("model_type not set, defaulting to binary")
                model_type <- "binary"
                
            }else{
                if(!model_type %in% model_types()[,1]){
                    stop("Error: 'model_type' not defined")
                }
            }
            
            # check act.fct is function
            if(!is.function(x[["act.fct"]])){
              if(is.character(x[["act.fct"]])){
                if(!x[["act.fct"]] %in% act_fcts()[,1]){
                  stop("Unrecognized activation function (act.fct).
                       Check options with act_fct().")
                }
                }else{
                  stop("Unrecognized activation function (act.fct)")
              }
            }
            
            # this allows a user to still provide a custom
            # function to easily test.  If the user must add the 
            # new function to the act_functions.hpp file for the
            # fast implementation.  Also update the act_fct() function.
            if(is.function(x[["act.fct"]])){
              result <- big.compute(x, covariate, rep, model_type)
            }else{
              result <- fast_compute(x, covariate, rep)
            }
            
           out <- new(paste0("nn_", model_type), 
                      neurons=result$neurons, 
                      net.result=result$net.result,
                      dvs = ifelse(is.null(x$model.list$response), 
                                   "unknown",
                                   x$model.list$response),
                      ivs = ifelse(is.null(x$model.list$variables),
                                   "unknown",
                                   x$model.list$variables)
           )
           return(out)  
          }
)

# ' @export
setMethod("compute", signature("fnn", "big.matrix"), 
          function(x, covariate, rep=1, model_type=NULL) {
              err_check <- check_nn_object(x)
#               if(!err_check){
#                   stop(err_check)
#               }
              
              if(is.null(model_type)){
                  warning("model_type not set, defaulting to binary")
                  model_type <- "binary"
                  
              }else{
                  if(!model_type %in% model_types()[,1]){
                      stop("Error: 'model_type' not defined")
                  }
              }
              
              # check act.fct is function
              if(!is.function(x[["act.fct"]])){
                  if(is.character(x[["act.fct"]])){
                      if(!x[["act.fct"]] %in% act_fcts()[,1]){
                          stop("Unrecognized activation function (act.fct).
                               Check options with act_fct().")
                      }
                      }else{
                          stop("Unrecognized activation function (act.fct)")
                  }
              }
              
              # this allows a user to still provide a custom
              # function to easily test.  If the user must add the 
              # new function to the act_functions.hpp file for the
              # fast implementation.  Also update the act_fct() function.
              if(is.function(x[["act.fct"]])){
                  result <- big.compute(x, covariate, rep, model_type)
              }else{
                  result <- fast_compute(x, covariate, rep)
              }
              
              out <- new(paste0("nn_", model_type), 
                         neurons=result$neurons, 
                         net.result=result$net.result,
                         dvs = ifelse(is.null(x$model.list$response), 
                                      "unknown",
                                      x$model.list$response),
                         ivs = ifelse(is.null(x$model.list$variables),
                                      "unknown",
                                      x$model.list$variables)
              )
              return(out)  
              }
)

# ' @export
setMethod("compute", signature("fnn", "data.frame"), 
          function(x, covariate, rep=1, model_type=NULL) {
              err_check <- check_nn_object(x)
#               if(!err_check){
#                   stop(err_check)
#               }
              
              if(is.null(model_type)){
                  warning("model_type not set, defaulting to binary")
                  model_type <- "binary"
              }else{
                  if(!model_type %in% model_types()[,1]){
                      stop("Error: 'model_type' not defined")
                  }
              }
              
              # check act.fct is function
              if(!is.function(x[["act.fct"]])){
                  if(is.character(x[["act.fct"]])){
                      if(!x[["act.fct"]] %in% act_fcts()[,1]){
                          stop("Unrecognized activation function (act.fct).
                               Check options with act_fct().")
                      }
                      }else{
                          stop("Unrecognized activation function (act.fct)")
                  }
              }
              
              # this allows a user to still provide a custom
              # function to easily test.  If the user must add the 
              # new function to the act_functions.hpp file for the
              # fast implementation.  Also update the act_fct() function.
              if(is.function(x[["act.fct"]])){
                  result <- neuralnet::compute(x, covariate, rep)
              }else{
                  result <- fast_compute(x, covariate, rep)
              }
              
              out <- new(paste0("nn_", model_type), 
                         neurons=result$neurons, 
                         net.result=result$net.result,
                         dvs = ifelse(is.null(x$model.list$response), 
                                      "unknown",
                                      x$model.list$response),
                         ivs = ifelse(is.null(x$model.list$variables),
                                      "unknown",
                                      x$model.list$variables)
              )
              return(out)  
              }
)

# ' @export
setMethod(
    "assign_categories", 
    signature("model_scores"), 
    function(object, dv=NULL){
        switch(class(object),
               payor_scores = {assign_categories_manual(object@x, 
                                                        model_type="payor")},
               binary_scores = {assign_categories_manual(object@x, 
                                                         model_type="binary",
                                                         dv = dv)},
               stop("object class not recognized")
        )
    })


# ' @export
setMethod(
    "assign_categories", 
    signature("model_subset"), 
    function(object, dv=NULL){
        switch(class(object),
               payor_subset = {assign_categories_manual(object@raw, 
                                                        model_type="payor")},
               binary_subset = {assign_categories_manual(object@raw, 
                                                         model_type="binary",
                                                         dv = dv)},
               stop("object class not recognized")
        )
    })


# ' @export
setMethod("assign_categories", signature("matrix"), 
          function(object, model_type=NULL, dv=NULL){
              switch(model_type,
                  payor = assign_categories_manual(object, model_type),
                  binary = assign_categories_manual(object, model_type, dv),
                  stop("object class not recognized")
              )
              
          })

# ' @export
setMethod("assign_categories", signature("big.matrix"), 
          function(object, model_type=NULL, dv=NULL){
              switch(model_type,
                     payor = assign_categories_manual(object, model_type),
                     binary = assign_categories_manual(object, model_type, dv),
                     stop("object class not recognized")
              )
          }
)

# ' @export
setMethod("assign_predicted_categories", signature("nn_model"), function(object){
  switch(class(object),
         nn_payor = {assign_predicted_payor_category(object@net.result)},
         nn_binary = {assign_predicted_binary_category(object@net.result)},
         stop("object class not recognized")
         )
})


# may change to model_scores object, will need switch statement for extracting specific columns
# ' @export
setMethod("assign_predicted_categories", signature("payor_scores"), function(object){
  # Need to select the appropriate columns
  # essentially 1+number of groups/categories
  switch(class(object),
         payor_scores = {assign_predicted_payor_category(object@x[,5:8])},
         stop("object class not recognized")
  )
})

# possibly change to model_subset to future model types
# ' @export
setMethod("assign_predicted_categories", signature("model_subset"), function(object){
  switch(class(object),
         payor_subset = {assign_predicted_payor_category(object@pred)},
         binary_subset = {assign_predicted_binary_category(object@pred)},
         stop("object class not recognized")
  )
})

# ' @export
setMethod("assign_scores_and_categories", 
          signature=c("nn_model", "matrixORbigmatrix"), 
          function(object, raw_x, ntiles=NULL, breaks=NULL, cutoffs=NULL, allowParallel=FALSE){
            switch(class(object),
                   nn_payor = {
                     assign_payor_scores_and_categories(object@net.result, 
                                                        raw_x, 
                                                        ntiles,
                                                        breaks, 
                                                        cutoffs, 
                                                        allowParallel
                     )},
                   nn_binary = {
                     assign_binary_scores_and_categories(object@net.result,
                                                         raw_x,
                                                         object@dvs,
                                                         ntiles,
                                                         breaks,
                                                         cutoffs,
                                                         allowParallel
                     )},
                   stop("object class not recognized")
            )
          })


setMethod("assign_predicted_categories", signature("model_scores"), function(object){
    switch(class(object),
           payor_scores = { stop("payor model not implemented for this function")
               #assign_predicted_payor_category(object@x[,2]))
               },
           binary_scores = {assign_predicted_binary_category(as.matrix(object@x[,2]))},
           stop("object class not recognized")
    )
})


setMethod("AUC",
          signature("model_scores"),
          function(object, dv=NULL){
              switch(class(object),
                     binary_scores = auc_binary(object@x, dv),
                     stop(paste0("The ", class(object), " class
                                 is not recognized"))
              )
          })


setMethod("AUC",
          signature("model_subset"),
          function(object){
              switch(class(object),
                     binary_subset = auc_sub_binary(object),
                     stop(paste0("The ", class(object), " class
                                 is not recognized"))
                     )
          })

# ' @export
setMethod("activation_scores_table",
          signature=c("nn_model", "matrixORbigmatrix"),
          function(object, raw_x, d_var=NULL, step=0.01, breaks=NULL, allowParallel=FALSE){
            switch(class(object),
                   nn_payor = {
                     payor_activation_scores_table(object@net.result, 
                                                   raw_x, 
                                                   d_var,
                                                   step, 
                                                   breaks,
                                                   allowParallel)
                   },
                   nn_binary = {
                     binary_activation_scores_table(object@net.result, 
                                                   raw_x,
                                                   d_var,
                                                   step, 
                                                   breaks,
                                                   allowParallel)
                   })
          })

#' @name score_subset,payor_scores-method
#' @title Subset Payor Model Score Categories
#' @description Subset the output from scores_and_categories to evaluate
#' prediction metrics above a given cutoff (e.g. 4, 7, etc.)
#' @param object The output from \code{\link{assign_scores_and_categories}}
#' @param cutoff The cutoff for filtering predictions (e.g. all those with confidence category >= 5)
#' @param category Optional argument to filter only on one payor category
#' @return A 'payor_subset' object
#' @return \describe{
#'  \item{pred}{Subset of the neuralnet activations}
#'  \item{raw}{Subset of the test data payor category columns}
#'  }
#' @seealso \code{\link{score_subset}}
# @rdname score_subset-methods
# @aliases score_subset,payor_scores-method
setMethod("score_subset", 
          signature("model_scores"), 
          function(object, cutoff, payor_category=NULL){
            switch(class(object),
                   payor_scores = {
                     payor_score_subset(object@x, cutoff, payor_category)
                   },
                   binary_scores = {
                     binary_score_subset(object@x, cutoff)
                   }
            )
          })

setMethod("percentile_profile",
          signature("model_scores"),
          function(object, step, allowParallel=FALSE){
            new_class = paste(unlist(strsplit(class(object), "_"))[1], 
                              "percentiles", sep="_")
            out <- percentile_profile_manual(object@x, 
                                             step, 
                                             allowParallel)
            new(new_class, percentiles = out)
          })

setMethod("classification_curve",
          signature("payor_percentiles"),
          function(object, filename){
            payor_classification_curve(object@percentiles, filename)
          })

setMethod("classification_curve",
          signature("binary_percentiles"),
          function(object, filename){
            binary_classification_curve(object@percentiles, filename)
          })

setMethod("lift_curve",
          signature("binary_percentiles"),
          function(object, data, dvs, filename){
            binary_lift_curve(object@percentiles, data, dvs, filename)
          })

setMethod("gains_chart",
          signature("binary_percentiles"),
          function(object, filename){
            binary_gains_chart(object@percentiles, filename)
          })

setMethod("demographic_profile",
          signature("model_scores"),
          function(object, data, dv_name, categories){
              switch(class(object),
                     binary_scores = demographic_profile_binary(object@x, 
                                                                data=data,
                                                                dv_name=dv_name,
                                                                categories=categories),
                     payor_scores = demographic_profile_payor(object@x, data=data),
                     stop("Unrecognized object, should be 'model_scores' object")
              )
            
          })


setMethod("extract_breakpoints", 
          signature=c("model_scores"), 
          function(object){
              switch(class(object),
                     payor_scores = {
                         stop("payor method not implemented")
                         },
                     binary_scores = {
                         extract_binary_breakpoints(object@x)
                         },
                     stop("object class not recognized")
              )
          })
