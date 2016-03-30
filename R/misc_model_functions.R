# Small function to return indices of a vector that are the maximum
maxidx <- function(arr) {
  return(which(arr == max(arr)))
}

# find indices of a vector within a given interval
#' @export
in_interval_int <- function(x, interval){
  stopifnot(length(interval) == 2L)
  which(interval[1] < x & x <= interval[2])
}

#' @title Assign Quantile Scores
#' @description This function computes scores defined by the confidence category,
#' user specified n-tiles
#' @param x An n x 16 matrix containing the raw activation values from 
#' @param ntiles The n-tiles for each payor category activation values
#' @param allowParallel Boolean indication if parallelization should be used
#' @return The 'x' matrix with the category columns filled
#' @export
assign_quantile_scores <- function(x, ntiles, allowParallel=FALSE){  
  # specify parallel
  `%op%` <- if(allowParallel){
    `%dopar%`
  }else{
    `%do%`
  }
  
  nc <- ncol(x)
  ng <- nc/4
  
  if(ng != 4 & ng != 1){
      if(nc %% 3 == 0){
          ng <- 1
      }else{
          stop("You must provide a dataframe with columns for at least the raw
           activation scores, quantile scores, and confidence categories")
      }
  }
  
  nr <- nrow(x)
  nt <- nrow(ntiles)
  
  idx <- foreach(z=seq(nt-1)) %:%
      foreach(y=seq(ng)) %op% {
          in_interval_int(x[,y+ng], ntiles[z:(z+1),y])
      }
  
  
  # remove % symbol and convert to numeric for scores
  scores <- as.numeric(gsub("%","",row.names(ntiles))) * 10

  for(i in seq(length(idx))){
      for(j in (ng*2+1):(ng*3)){
          cur_idx <- idx[[i]][[j-(ng*2)]]
          x[cur_idx, j] <- scores[i]
      }
  }
  
  
  idx_max <- list()
  idx_min <- list()
  for(i in seq(ng)){
      idx_max[i] <- list(which(x[,i+ng] >= ntiles[nrow(ntiles)]))
      idx_min[i] <- list(which(x[,i+ng] <= ntiles[1]))
  }
  
  
  # fill the remaining max cases with 999 and min with 0
  for(i in seq(ng)){
      x[idx_max[[i]], i+ng+1] <- 999
      x[idx_min[[i]], i+ng+1] <- 0
  }
  
  
#   # fill the remaining max cases with 999
#   idx_max <- lapply(seq(ng), function(y) which(!seq(nr) %in% unlist(lapply(idx, function(x) x[[y]]))))
#   idx_min <- lapply(seq(ng), function(y) which(!seq(nr) %in% unlist(lapply(idx, function(x) x[[y]]))))
#   
#   for(i in seq(ng)){
#       x[idx_max[[i]], i+(ng*2)] <- 999
#   }
  
  return(x)  
}

#' @title Assign Manual Quantile Scores
#' @description This function computes scores defined by provided cut points.
#' This functions requires the user to provide the breakpoints to manually
#' calculate the scores.  This is representative of the original approach to
#' calculating scores.
#' @param x An n x 3 matrix containing the raw activation values from 
#' @param ntiles The n-tiles for each confidence category activation values
#' @param allowParallel Boolean indication if parallelization should be used
#' @return The 'x' matrix with the score columns filled
#' @export
assign_manual_quantile_scores <- function(x, ntiles, allowParallel=FALSE){  
    # specify parallel
    `%op%` <- if(allowParallel){
        `%dopar%`
    }else{
        `%do%`
    }
    
    assert_is_not_null(ntiles)    
    if(!is.matrix(ntiles)){
        assert_is_of_length(ntiles, 8L)
        ntiles <- matrix(ntiles, ncol=1)
    }
    
    nc <- ncol(x)
    if(nc > 3){
        stop("Function only valid for matrices with 3 columns")
    }
    
    ng <- 1
    nr <- nrow(x)
    nt <- nrow(ntiles)
    score_bases <- c(560,810,910,960,985,995,1000)
    
    idx <- foreach(z=seq(nt-1)) %op% {
        in_interval_int(x[,ng], ntiles[z:(z+1),1])
    }
    
    
    for(i in seq(from=1, to=length(idx)-1)){
        cur_idx <- idx[[i]]
        x[cur_idx, 2] <- floor(score_bases[i] * (x[cur_idx,1] - ntiles[i])/(ntiles[i+1] - ntiles[i]))
    }

    # fill the remaining max cases with 999
    x[x$score < 0 ,2] <- 0
    x[x$score > 999, 2] <- 999
    
    return(x)     
}


#' @title Assign Quantile Categories
#' @description This function assigns confidence categories defined by the 
#' user specified n-tiles
#' @param x A matrix containing at least 3 columns corresponding to 'raw',
#' 'score' and 'cat' columns.
#' @param cutoffs A numeric vector of length 8, specifying the cutoffs for
#' the scores to correspond to the respective category (1-7).
#' @return The 'x' matrix with the category columns filled.
#' @export
assign_quantile_category <- function(x, cutoffs=NULL){  
  
  # set vector for categories
  categories <- seq(7)
  # set vector for score cutoffs
  if(is.null(cutoffs)){
    cutoffs <- c(0,560,810,910,960,985,995,999)
  }else{
    stopifnot(length(cutoffs) == 8L)
    assert_all_are_positive(cutoffs)
    if(is.numeric(cutoffs) | is.integer(cutoffs)){
      stop("'cutoffs' is not of type 'integer' or 'numeric'")
    }
  }
  
  nc <- ncol(x)
  ng <- nc/4
  
  if(ng != 4 & ng != 1){
    if(nc %% 3 == 0){
      ng <- 1
    }else{
      stop("You must provide a dataframe with columns for at least the raw
           activation scores, quantile scores, and confidence categories")
    }
  }
  
  
  # would have preferred to have list of matrices (sapply) for smaller size but 
  # different lengths of indices are possible and cause errors
  # possible place for if else condition
  if(nc == 3){
      idx <- lapply(categories, function(z) 
              in_interval_int(x[,2], cutoffs[z:(z+1)]))
  }else{
      idx <- lapply(categories, function(z) 
          lapply(seq(ng), function(y) 
              in_interval_int(x[,y+(ng*2)], cutoffs[z:(z+1)]))) 
  }

  if(nc == 3){
      for(i in categories){
          for(j in nc){
              cur_idx <- idx[[i]]
              x[cur_idx, j] <- categories[i]
          }
      }
  }else{
      for(i in categories){
          for(j in (nc-ng+1):nc){
              cur_idx <- idx[[i]][[j-(nc-ng)]]
              x[cur_idx, j] <- categories[i]
          }
      }
  }

  
  out <- as.matrix(x)
  return(out)
}


#' @title Probability Classification Profile
#' @description This function calculates the probability of the group
#' being accurately predicted across user defined bins of length \code{step}.
#' This provide a matrix of values that can be used to rapidly create a lift
#' curve with the \code{\link{lift_curve}} function.
#' @param x The output from \code{\link{assign_scores_and_categories}}
#' @param step The width of the bins created from 0 to 1
#' @param allowParallel Optional argument to utilize parallel processing
#' @return A matrix contining the probabilities and associated quantile
#' cutoffs
percentile_profile_manual <- function(x, step, allowParallel=FALSE){
  
  assert_is_logical(allowParallel)
  if(missing(step)){
    stop("The argument 'step' is missing")
  }
  
  # specify parallel
  `%op%` <- if(allowParallel){
    `%dopar%`
  }else{
    `%do%`
  }
  
  # placeholders to avoid repetition and allow alternate numbers 
  # of columns for different models
  nc <- ncol(x)
  ng <- nc/4
  nr <- nrow(x)
  nch <- nchar(step)
  end <- as.numeric(substr(as.character(.999999), 1, nch))
  steps <- c(seq(step,end,step)*1000, 1000)
#   steps <- c(seq(step,end,step)*1000)
  pp <- step*1000
  
  # parallelized
  out_mat <- foreach(g = seq(ng), .combine="cbind") %:%
    foreach(p = seq(length(steps)), .combine="c") %op% {
      pn <- (1000-steps[p])
      sum(x[x[,(g+ng*2)] >= pn & x[,(g+ng*2)] < pn+pp, g])/(nr*step)
    }
  
  # add 0 base
  if(!is.matrix(out_mat)){
    out_mat <- matrix(out_mat, ncol=1)
  }
  
#     out_mat <- rbind(out_mat, 0)
  
  # remove any 'errors' above 1
  if(any(out_mat > 1)){
    out_mat[out_mat>1] = 1
  }
  
  out_mat <- cbind(out_mat, sort(c(steps)/1000, decreasing=TRUE))
#   out_mat <- cbind(out_mat, sort(steps/1000, decreasing=TRUE))
  colnames(out_mat) <- NULL
  return(out_mat)
}


percentile_profile_manual2 <- function(x, step, allowParallel=FALSE){
  
  assert_is_logical(allowParallel)
  if(missing(step)){
    stop("The argument 'step' is missing")
  }
  
  # specify parallel
  `%op%` <- if(allowParallel){
    `%dopar%`
  }else{
    `%do%`
  }
  
  # placeholders to avoid repetition and allow alternate numbers 
  # of columns for different models
  nc <- ncol(x)
  ng <- nc/4
  nr <- nrow(x)
  nch <- nchar(step)
  end <- as.numeric(substr(as.character(.999999), 1, nch))
  steps <- c(seq(step,end,step)*1000, 1000)
  pp <- step*1000
  
  # parallelized
  # this seems more logical, percent of predicted categories
  # accurately predicted given the specific cutoff
  out_mat <- foreach(g = seq(ng), .combine="cbind") %:%
    foreach(p = seq(length(steps)), .combine="c") %op% {
      pn <- (1000-steps[p])
      idx <- x[,(g+ng*2)] >= pn & x[,(g+ng*2)] < pn+pp
      1 - sum(abs(round(x[idx, g+ng]) - x[idx, g])) / length(idx)
    }
  
  # add between 0 and lowest step
  out_mat <- c(out_mat, 0)
  
  # remove any 'errors' above 1
  if(any(out_mat > 1)){
    out_mat[out_mat>1] = 1
  }
  
  out_mat <- cbind(out_mat, sort(c(0, steps)/1000, decreasing=TRUE))
  colnames(out_mat) <- NULL
  return(out_mat)
}

demographic_profile_indv <- function(mat, data){
    
    # we currently don't care about individuals so rename rows for ease-of-use
    #if(is.null(row.names(mat))){
    row.names(mat) <- seq(nrow(mat))
    #}
    
    # must use 'split.data.frame' to prevent 'split' from returning vectors
    idx <- lapply(split.data.frame(mat, mat[,ncol(mat)]), function(x) as.numeric(row.names(x)))
    mat_split <- lapply(idx, function(x) data[c(x),])
    
    if(any(!factor(seq(7)) %in% names(mat_split))){
        missing.factors <- seq(7)[which(!factor(seq(7)) %in% names(mat_split))]
        for(i in missing.factors){
            # blank named vector
            filler <- matrix(rep(0, 10), nrow=1)
            colnames(filler) <- colnames(mat_split[[1]])
            
            # append to list in proper spot
            mat_split <- append(mat_split, list(filler), after=i-1)
        }
        names(mat_split) <- as.character(seq(7))
        warning("Some confidence categories were not present for some groups", call.=FALSE)
    }
    
    mat_demog <- lapply(mat_split, colSums)
    mat_out <- do.call("cbind", mat_demog)
    mat_n <- lapply(mat_split, nrow)
    mat_n <- sapply(mat_n, function(x) ifelse(x==1, 0, x))
    mat_out <- rbind(mat_n, mat_out)
    return(mat_out)
}

demographic_profile_indv2 <- function(mat, data){
    
    # we currently don't care about individuals so rename rows for ease-of-use
    row.names(mat) <- seq(nrow(mat))
    
    # must use 'split.data.frame' to prevent 'split' from returning vectors
    idx <- lapply(split.data.frame(mat, mat[,ncol(mat)]), function(x) as.numeric(row.names(x)))
    mat_split <- lapply(idx, function(x) data[c(x),])
    
    if(any(!factor(seq(999)) %in% names(mat_split))){
        missing.factors <- seq(999)[which(!factor(seq(999)) %in% names(mat_split))]
        for(i in missing.factors){
            # blank named vector
            filler <- matrix(rep(0, 10), nrow=1)
            colnames(filler) <- colnames(mat_split[[1]])
            
            # append to list in proper spot
            mat_split <- append(mat_split, list(filler), after=i-1)
        }
        names(mat_split) <- as.character(seq(999))
        warning("Some confidence categories were not present for some groups", call.=FALSE)
    }
    
    mat_demog <- lapply(mat_split, colSums)
    mat_out <- do.call("cbind", mat_demog)
    mat_n <- lapply(mat_split, nrow)
    mat_n <- sapply(mat_n, function(x) ifelse(x==1, 0, x))
    mat_out <- rbind(mat_n, mat_out)
    return(mat_out)
}

