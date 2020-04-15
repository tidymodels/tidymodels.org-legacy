pkgs <- c("AmesHousing", "caret", "devtools", "discrim", "doParallel", "DT", 
          "embed", "forecast", "fs", "furrr", "ggiraph", "glmnet", "glue", 
          "keras", "kernlab", "klaR", "mlbench", "modeldata", "pls", "randomForest", 
          "ranger", "rlang", "scales", "stringr", "survivial", "sweep", 
          "textfeatures", "textrecipes", "themis", "tidymodels", "tidyposterior", 
          "timetk", "zoo")

install.packages(pkgs, repos = "https://cran.rstudio.com")

library(devtools)

install_github("therneau/survival")
install_github("tidymodels/tidymodels")

library(keras)
install_keras(method = "virtualenv")

