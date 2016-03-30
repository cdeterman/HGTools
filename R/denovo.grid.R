

#' @title Denovo Grid Generation
#' @description Greates grid for optimizing selected models
#' 
#' @param data data of method to be tuned
#' @param method vector indicating the models to generate grids.
#' Available options are \code{"neuralnet"} (Neural Network), 
#' \code{"rf"} (Random Forest), \code{"gbm"} (Gradient Boosting Machine),
#'  \code{"svm"} (Support Vector Machines), and \code{"glmnet"} 
#'  (Elastic-net Generalized Linear Model)
#' @param res Resolution of model optimization grid.
#' @return A list containing dataframes of all combinations of 
#' parameters for each model:
#' @author Charles Determan Jr
#' @seealso \code{"expand.grid"} for generating grids of specific 
#' parameters desired.  However, NOTE that you must still convert 
#' the generated grid to a list.
#' @export
denovo.grid <- 
    function(data,
             method,
             res
    )
    {
        assert_is_data.frame(data)
        assert_is_character(method)
        assert_is_numeric(res)
        
        if(!".classes" %in% colnames(data)){
            stop("Creating a tuning grid requires a '.classes' column
                 representing the class/group labels.")
        }
        
        # number of columns - 1
        nc <- dim(data)[2] - 1
        
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
                   
                   rf = rfTune(data, res),
                   
                   svm = data.frame(.C = 2 ^((1:res) - 3)),
                   
                   stop("unrecognized method")
            )   
        return(out)
    }


rfTune <- function(
    data,
    res
)
{
    p <- dim(data)[2] - 1 
    c <- ceiling(p/50)
    
    # sequence of trees to try
    # both for high and 'very high' dimensional data
    if(p < 500 ){
        if(p < 100){
            treeSeq <- floor(seq(1, to = p, length = res))    
        }else{
            treeSeq <- floor(seq(10, to = p, length = c))
        }
    }else{
        treeSeq <- floor(2^seq(5, to = log(p, base = 2), length = c))
    } 
    
    # check if any of the numbers are repeated (i.e. repeating the 
    # same number of trees is inefficient)
    if(any(table(treeSeq) > 1))
    {
        treeSeq <- unique(treeSeq)
    }
    data.frame(.mtry = treeSeq)
}
