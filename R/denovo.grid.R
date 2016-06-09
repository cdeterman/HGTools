
#' @title Denovo Grid Generation
#' @description Greates grid for optimizing selected models
#' @param method vector indicating the models to generate grids.
#' Available options are \code{"neuralnet"} (Neural Network), 
#' \code{"rf"} (Random Forest), \code{"gbm"} (Gradient Boosting Machine),
#'  \code{"svm"} (Support Vector Machines), and \code{"glmnet"} 
#'  (Elastic-net Generalized Linear Model)
#' @param res Resolution of model optimization grid.
#' @param data data of method to be tuned
#' @param dvs character vector identifying the dependent variables in the dataset
#' @return A list containing dataframes of all combinations of 
#' parameters for each model:
#' @author Charles Determan Jr
#' @seealso \code{"expand.grid"} for generating grids of specific 
#' parameters desired.  However, NOTE that you must still use the same
#' arguments names prefixed with a '.'.
#' @import assertive
#' @import hgneuralnet
#' @export
denovo.grid <- 
    function(method,
             res,
             data,
             dvs = NULL
    )
    {
        assert_is_character(method)
        assert_is_numeric(res)
        
        if("gbm" %in% method) {
            .tree.options = c(500, 1000, 2000, 5000, 10000)
        }
        
        out <-
            switch(tolower(method),
                   neuralnet = expand.grid(
                       .hidden = seq(from = 2, to = 2 + res),
                       .threshold = seq(from = 5, to = 1, length = res)),
                   # alpha always tuned
                   # lambda tuning depends on user decision
                   #   if defined number of features (f), cannot be tuned
                   #   if doing full rank correlations 
                   #       (i.e. f = nc), cannot be tuned
                   #   if letting glmnet decide optimal features, then tune
                   glmnet = expand.grid(
                       .alpha = seq(0.1, 1, length = res),
                       .lambda = seq(.1, 3, length = 3 * res)),
                   
                   # n.trees doesn't need to be looped because can access 
                   # any prior number from max
                   # modified .shrinkage from just '.1' to sequence 
                   # of values
                   gbm = expand.grid(
                       .interaction.depth = seq(1, res),
                       .n.trees = .tree.options[1:res],                    
                       .shrinkage = c(.1/seq(res))),
                   
                   rf = rfTune(data, dvs, res),
                   
                   svm = data.frame(.C = 2 ^((1:res) - 3)),
                   
                   stop("unrecognized method")
            )   
        return(out)
    }


rfTune <- function(
    data,
    dvs,
    res
)
{
    assert_is_not_null(data)
    
    if(is.null(dvs)){
        warning("RandomForest uses the data dimensions to generate a grid. 
                Make sure this your intention or provide the dvs.")
    }
    
    p <- dim(data)[2] - ifelse(is.null(dvs), 0, length(dvs)) 
    
    # sequence of trees to try
    # both for high and 'very high' dimensional data
    if(p < 500 ){
        treeSeq <- floor(seq(2, to = p, length = res)) 
    }else{
        treeSeq <- floor(2^seq(5, to = log(p, base = 2), length = res))
    } 
    
    # check if any of the numbers are repeated (i.e. repeating the 
    # same number of trees is inefficient)
    if(any(table(treeSeq) > 1))
    {
        treeSeq <- unique(treeSeq)
    }
    data.frame(.mtry = treeSeq)
}


#' @importFrom hgneuralnet act_fcts

#' @title Denovo Grid Generation
#' @description Greates expanded grid for neuralnets
#' @param res Resolution of the search grid
#' @param act_fcts A character vector of activation functions
#' @param dropout A boolean indicating if dropout iterations should be included
#' @author Charles Determan Jr
#' @seealso \code{\link{denovo.grid}} or \code{\link{"expand.grid"}} for generating grids 
#' of specific parameters desired.  However, NOTE that you must still use the same
#' arguments names prefixed with a '.'.
#' @import assertive
#' @export
denovo_neuralnet_grid <- 
    function(res, act_fcts, dropout){
        
        assert_is_character(act_fcts)
        
        if(any(!act_fcts %in% act_fcts()$act_fcts)){
            stop("Activation function provided not implemented")
        }
        
        grid <- expand.grid(
            .hidden = seq(from = 2, to = 2 + res),
            .threshold = seq(from = 5, to = 1, length = res),
            .act_fcts = act_fcts,
            .dropout = dropout,
            .visible_dropout = if(dropout){seq(from = 0, to=0.2, length = res)}else{0},
            .hidden_dropout = if(dropout){seq(from=0.1, to=0.5, length=res)}else{0})
        
        
        return(grid)
    }
