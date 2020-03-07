req_pkgs <- function(x) {
  x <- sort(x)
  x <- paste0("`", x, "`")
  x <- knitr::combine_words(x, and = " and ")
  paste0(
    "This article requires the following packages to be installed: ",
    x, "."
  )
}
