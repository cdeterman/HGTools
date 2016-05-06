check_nn_object <- function(object) {
  errors <- character()
  
  print(names(object))
  
  # check components present
  elements <- c("weights", "act.fct", "linear.output")
  if(!all(elements %in% names(object))){
    idx <- which(!all(elements %in% names(object)))
    msg <- paste("x does not contain required elements.  Missing ", elements[idx], sep="")
    errors <- c(errors, msg)
  }
  
  # removed this error to accomodate the Rcpp implementations
  # the check applied in the generic to determine the appropriate
  # function to call
  
#   # check act.fct is function
#   if(!is.function(object[["act.fct"]])){
#     msg <- paste("act.fct is not a function")
#     errors <- c(errors, msg)
#   }
  
  # check that linear.output is logical
  if(!is.logical(object[["linear.output"]])){
    msg <- paste("act.fct is not logical")
    errors <- c(errors, msg)
  }
  
  if(length(errors) == 0) TRUE else stop(errors)
}