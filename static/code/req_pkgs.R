req_pkgs <- function(x, what = "This article") {
  x <- sort(x)
  x <- paste0("`", x, "`")
  x <- knitr::combine_words(x, and = " and ")
  paste0(
    what,
    " requires that you have the following packages installed: ",
    x, "." 
  )
}
