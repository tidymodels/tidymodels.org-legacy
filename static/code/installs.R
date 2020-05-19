install.packages("remotes", repos = "https://cran.rstudio.com")

pkg_list <- read.dcf("DESCRIPTION")[,"Imports"]
pkg_list <- gsub("\n", "", pkg_list, fixed = TRUE)
pkg_list <- strsplit(pkg_list, ",")[[1]]

remotes::install_cran(
  pkg_list,
  repos = "https://cran.rstudio.com",
  upgrade = "alwyas",
  type = "source",
  force = TRUE
)

library(remotes)

install_github("rstudio/DT")

library(keras)

install_keras(method = "virtualenv")

