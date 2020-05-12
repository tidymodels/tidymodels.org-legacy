pkg_list <- read.dcf("DESCRIPTION")[,"Imports"]
pkg_list <- gsub("\n", "", pkg_list, fixed = TRUE)
pkg_list <- strsplit(pkg_list, ",")[[1]]

install.packages(pkg_list, repos = "https://cran.rstudio.com")

library(devtools)

install_github("rstudio/DT")

library(keras)

install_keras(method = "virtualenv")

