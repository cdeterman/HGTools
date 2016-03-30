#' @title Read SQL file
#' @description Reads a .sql file and creates a character vector to be used in a dbSendQuery().
#' @param dir Location of the .sql file
#' @export

read.sql <- function(dir){
  query <- readChar(dir, file.info(dir)$size)
  query <- gsub("/\\*.*?\\*/", "", gsub("--.*?\n", "", query))
  query <- gsub("\r?\n", " ", query)
  return(query)
}