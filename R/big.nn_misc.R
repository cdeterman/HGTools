
differentiate <-
  function (
    orig.fct, 
    hessian = FALSE) 
  {
    body.fct <- deparse(body(orig.fct))
    if (body.fct[1] == "{") 
      body.fct <- body.fct[2]
    text <- paste("y~", body.fct, sep = "")
    text2 <- paste(deparse(orig.fct)[1], "{}")
    temp <- deriv(eval(parse(text = text)), "x", func = eval(parse(text = text2)), 
                  hessian = hessian)
    temp <- deparse(temp)
    derivative <- NULL
    if (!hessian) 
      for (i in 1:length(temp)) {
        if (!any(grep("value", temp[i]))) 
          derivative <- c(derivative, temp[i])
      }
    else for (i in 1:length(temp)) {
      if (!any(grep("value", temp[i]), grep("grad", temp[i]), 
               grep(", c", temp[i]))) 
        derivative <- c(derivative, temp[i])
    }
    number <- NULL
    for (i in 1:length(derivative)) {
      if (any(grep("<-", derivative[i]))) 
        number <- i
    }
    if (is.null(number)) {
      return(function(x) {
        matrix(0, nrow(x), ncol(x))
      })
    }
    else {
      derivative[number] <- unlist(strsplit(derivative[number], 
                                            "<-"))[2]
      derivative <- eval(parse(text = derivative))
    }
    if (length(formals(derivative)) == 1 && length(derivative(c(1, 
                                                                1))) == 1) 
      derivative <- eval(parse(text = paste("function(x){matrix(", 
                                            derivative(1), ", nrow(x), ncol(x))}")))
    if (length(formals(derivative)) == 2 && length(derivative(c(1, 
                                                                1), c(1, 1))) == 1) 
      derivative <- eval(parse(text = paste("function(x, y){matrix(", 
                                            derivative(1, 1), ", nrow(x), ncol(x))}")))
    return(derivative)
  }


# display function to report status
display <-
  function (hidden, threshold, rep, i.rep, lifesign) 
  {
    text <- paste("    rep: %", nchar(rep) - nchar(i.rep), "s", 
                  sep = "")
    cat("hidden: ", paste(hidden, collapse = ", "), "    thresh: ", 
        threshold, sprintf(eval(expression(text)), ""), i.rep, 
        "/", rep, "    steps: ", sep = "")
    if (lifesign == "full") 
      lifesign <- sum(nchar(hidden)) + 2 * length(hidden) - 
      2 + max(nchar(threshold)) + 2 * nchar(rep) + 41
    return(lifesign)
  }


generate.rownames <-
  function (matrix, weights, model.list) 
  {
    rownames <- rownames(matrix)[rownames(matrix) != ""]
    for (w in 1:length(weights)) {
      for (j in 1:ncol(weights[[w]])) {
        for (i in 1:nrow(weights[[w]])) {
          if (i == 1) {
            if (w == length(weights)) {
              rownames <- c(rownames, paste("Intercept.to.", 
                                            model.list$response[j], sep = ""))
            }
            else {
              rownames <- c(rownames, paste("Intercept.to.", 
                                            w, "layhid", j, sep = ""))
            }
          }
          else {
            if (w == 1) {
              if (w == length(weights)) {
                rownames <- c(rownames, paste(model.list$variables[i - 
                                                                     1], ".to.", model.list$response[j], sep = ""))
              }
              else {
                rownames <- c(rownames, paste(model.list$variables[i - 
                                                                     1], ".to.1layhid", j, sep = ""))
              }
            }
            else {
              if (w == length(weights)) {
                rownames <- c(rownames, paste(w - 1, "layhid.", 
                                              i - 1, ".to.", model.list$response[j], 
                                              sep = ""))
              }
              else {
                rownames <- c(rownames, paste(w - 1, "layhid.", 
                                              i - 1, ".to.", w, "layhid", j, sep = ""))
              }
            }
          }
        }
      }
    }
    rownames(matrix) <- rownames
    colnames(matrix) <- 1:(ncol(matrix))
    return(matrix)
  }


relist <-
  function (x, nrow, ncol) 
  {
    list.x <- NULL
    for (w in 1:length(nrow)) {
      length <- nrow[w] * ncol[w]
      list.x[[w]] <- matrix(x[1:length], nrow = nrow[w], ncol = ncol[w])
      x <- x[-(1:length)]
    }
    list.x
  }

remove.intercept <- function(x) UseMethod("remove.intercept")

remove.intercept.default <-
  function (x) 
  {
    matrix(x[-1, ], ncol = ncol(x))
  }

remove.intercept.big.matrix <-
  function (x) 
  {
    #sub.big.matrix(x, firstCol = 2)
    deepcopy(x, cols=2:ncol(x), shared=FALSE)
  }


type <-
  function (fct) 
  {
    attr(fct, "type")
  }


print.nn <-
  function (x, ...) 
  {
    matrix <- x$result.matrix
    cat("Call: ", deparse(x$call), "\n\n", sep = "")
    if (!is.null(matrix)) {
      if (ncol(matrix) > 1) {
        cat(ncol(matrix), " repetitions were calculated.\n\n", 
            sep = "")
        sorted.matrix <- matrix[, order(matrix["error", ])]
        if (any(rownames(sorted.matrix) == "aic")) {
          print(t(rbind(Error = sorted.matrix["error", 
                                              ], AIC = sorted.matrix["aic", ], BIC = sorted.matrix["bic", 
                                                                                                   ], `Reached Threshold` = sorted.matrix["reached.threshold", 
                                                                                                                                          ], Steps = sorted.matrix["steps", ])))
        }
        else {
          print(t(rbind(Error = sorted.matrix["error", 
                                              ], `Reached Threshold` = sorted.matrix["reached.threshold", 
                                                                                     ], Steps = sorted.matrix["steps", ])))
        }
      }
      else {
        cat(ncol(matrix), " repetition was calculated.\n\n", 
            sep = "")
        if (any(rownames(matrix) == "aic")) {
          print(t(matrix(c(matrix["error", ], matrix["aic", 
                                                     ], matrix["bic", ], matrix["reached.threshold", 
                                                                                ], matrix["steps", ]), dimnames = list(c("Error", 
                                                                                                                         "AIC", "BIC", "Reached Threshold", "Steps"), 
                                                                                                                       c(1)))))
        }
        else {
          print(t(matrix(c(matrix["error", ], matrix["reached.threshold", 
                                                     ], matrix["steps", ]), dimnames = list(c("Error", 
                                                                                              "Reached Threshold", "Steps"), c(1)))))
        }
      }
    }
    cat("\n")
  }
