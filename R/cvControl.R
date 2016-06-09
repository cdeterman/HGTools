#' @title Control function for 'train'
#' @description This function provides a means to consolidate additional
#' parameters for the train function to avoid verbose input parameters.
#' @param model_args A named list of additional arguements for the internal
#' method called.
#' @param filter Boolean argument indicating if prediction results should
#' be filtered
#' @param filter_category A numeric value for the category filter if \code{filter}
#' is \code{TRUE}
#' @param verbose A boolean option for verbose model output
#' @param scale A boolean option whether to scale model results
#' cross-validation step which can be subsequently loaded with \code{load} 
#' @return A list of the additional internal parameters
#' @import assertive
#' @export
cvControl <- function(model_args, model_type, 
                      filter = FALSE, filter_category = NULL, 
                      verbose = FALSE,
                      scale = FALSE){
    if(!missing(model_args)){
        assert_is_list(model_args)
    }else{
        model_args <- NULL
    }
    
    if(!missing(model_type)){
        assert_is_character(model_type)
    }else{
        model_type <- NULL
    }
    
    assert_is_logical(verbose)
    assert_is_logical(scale)
    
    if(filter){
        assert_is_non_empty(filter_category)
        assert_is_not_null(filter_category)
        assert_all_are_in_closed_range(filter_category, lower = 4, upper = 7)
    }
    
    out <- list(model_args=model_args,
                model_type=model_type,
                filter = list(filter = filter, 
                              filter_category = if(filter) filter_category else NULL),
                verbose = verbose,
                scale = scale)
    return(out)
}
