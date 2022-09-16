---
title: "Classification models using a neural network"
tags: [rsample, parsnip]
categories: [model fitting]
type: learn-subsection
weight: 2
description: | 
  Train a classification model and evaluate its performance.
---


  



## Introduction

To use the code in this article, you will need to install the following packages: keras and tidymodels. You will also need the python keras library installed (see `?keras::install_keras()`).

We can create classification models with the tidymodels package [parsnip](https://parsnip.tidymodels.org/) to predict categorical quantities or class labels. Here, let's fit a single classification model using a neural network and evaluate using a validation set. While the [tune](https://tune.tidymodels.org/) package has functionality to also do this, the parsnip package is the center of attention in this article so that we can better understand its usage. 

## Fitting a neural network


Let's fit a model to a small, two predictor classification data set. The data are in the modeldata package (part of tidymodels) and have been split into training, validation, and test data sets. In this analysis, the test set is left untouched; this article tries to emulate a good data usage methodology where the test set would only be evaluated once at the end after a variety of models have been considered. 



```r
data(bivariate)
nrow(bivariate_train)
#> [1] 1009
nrow(bivariate_val)
#> [1] 300
```

A plot of the data shows two right-skewed predictors: 


```r
ggplot(bivariate_train, aes(x = A, y = B, col = Class)) + 
  geom_point(alpha = .2)
```

<img src="figs/biv-plot-1.svg" width="576" />

Let's use a single hidden layer neural network to predict the outcome. To do this, we transform the predictor columns to be more symmetric (via the `step_BoxCox()` function) and on a common scale (using `step_normalize()`). We can use [recipes](https://recipes.tidymodels.org/) to do so:


```r
biv_rec <- 
  recipe(Class ~ ., data = bivariate_train) %>%
  step_BoxCox(all_predictors())%>%
  step_normalize(all_predictors()) %>%
  prep(training = bivariate_train, retain = TRUE)

# We will bake(new_data = NULL) to get the processed training set back

# For validation:
val_normalized <- bake(biv_rec, new_data = bivariate_val, all_predictors())
# For testing when we arrive at a final model: 
test_normalized <- bake(biv_rec, new_data = bivariate_test, all_predictors())
```

We can use the keras package to fit a model with 5 hidden units and a 10% dropout rate, to regularize the model:


```r
set.seed(57974)
nnet_fit <-
  mlp(epochs = 100, hidden_units = 5, dropout = 0.1) %>%
  set_mode("classification") %>% 
  # Also set engine-specific `verbose` argument to prevent logging the results: 
  set_engine("keras", verbose = 0) %>%
  fit(Class ~ ., data = bake(biv_rec, new_data = NULL))
#> Loaded Tensorflow version 2.9.2

nnet_fit
#> parsnip model object
#> 
#> Model: "sequential"
#> ________________________________________________________________________________
#>  Layer (type)                       Output Shape                    Param #     
#> ================================================================================
#>  dense (Dense)                      (None, 5)                       15          
#>  dense_1 (Dense)                    (None, 5)                       30          
#>  dropout (Dropout)                  (None, 5)                       0           
#>  dense_2 (Dense)                    (None, 2)                       12          
#> ================================================================================
#> Total params: 57
#> Trainable params: 57
#> Non-trainable params: 0
#> ________________________________________________________________________________
```

## Model performance

In parsnip, the `predict()` function can be used to characterize performance on the validation set. Since parsnip always produces tibble outputs, these can just be column bound to the original data: 


```r
val_results <- 
  bivariate_val %>%
  bind_cols(
    predict(nnet_fit, new_data = val_normalized),
    predict(nnet_fit, new_data = val_normalized, type = "prob")
  )
val_results %>% slice(1:5)
#> # A tibble: 5 × 6
#>       A     B Class .pred_class .pred_One .pred_Two
#>   <dbl> <dbl> <fct> <fct>           <dbl>     <dbl>
#> 1 1061.  74.5 One   Two             0.417    0.583 
#> 2 1241.  83.4 One   Two             0.434    0.566 
#> 3  939.  71.9 One   One             0.776    0.224 
#> 4  813.  77.1 One   One             0.983    0.0173
#> 5 1706.  92.8 Two   Two             0.215    0.785

val_results %>% roc_auc(truth = Class, .pred_One)
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.815

val_results %>% accuracy(truth = Class, .pred_class)
#> # A tibble: 1 × 3
#>   .metric  .estimator .estimate
#>   <chr>    <chr>          <dbl>
#> 1 accuracy binary         0.733

val_results %>% conf_mat(truth = Class, .pred_class)
#>           Truth
#> Prediction One Two
#>        One 149  27
#>        Two  53  71
```

Let's also create a grid to get a visual sense of the class boundary for the validation set.


```r
a_rng <- range(bivariate_train$A)
b_rng <- range(bivariate_train$B)
x_grid <-
  expand.grid(A = seq(a_rng[1], a_rng[2], length.out = 100),
              B = seq(b_rng[1], b_rng[2], length.out = 100))
x_grid_trans <- bake(biv_rec, x_grid)

# Make predictions using the transformed predictors but 
# attach them to the predictors in the original units: 
x_grid <- 
  x_grid %>% 
  bind_cols(predict(nnet_fit, x_grid_trans, type = "prob"))

ggplot(x_grid, aes(x = A, y = B)) + 
  geom_contour(aes(z = .pred_One), breaks = .5, col = "black") + 
  geom_point(data = bivariate_val, aes(col = Class), alpha = 0.3)
```

<img src="figs/biv-boundary-1.svg" width="576" />



## Session information


```
#> ─ Session info ─────────────────────────────────────────────────────
#>  setting  value
#>  version  R version 4.2.1 (2022-06-23)
#>  os       macOS Big Sur ... 10.16
#>  system   x86_64, darwin17.0
#>  ui       X11
#>  language (EN)
#>  collate  en_US.UTF-8
#>  ctype    en_US.UTF-8
#>  tz       America/Los_Angeles
#>  date     2022-09-16
#>  pandoc   2.17.1.1 @ /Applications/RStudio.app/Contents/MacOS/quarto/bin/ (via rmarkdown)
#> 
#> ─ Packages ─────────────────────────────────────────────────────────
#>  package    * version date (UTC) lib source
#>  broom      * 1.0.1   2022-08-29 [1] CRAN (R 4.2.0)
#>  dials      * 1.0.0   2022-06-14 [1] CRAN (R 4.2.0)
#>  dplyr      * 1.0.10  2022-09-01 [1] CRAN (R 4.2.0)
#>  ggplot2    * 3.3.6   2022-05-03 [1] CRAN (R 4.2.0)
#>  infer      * 1.0.3   2022-08-22 [1] CRAN (R 4.2.0)
#>  keras        2.9.0   2022-05-23 [1] CRAN (R 4.2.0)
#>  parsnip    * 1.0.1   2022-08-18 [1] CRAN (R 4.2.0)
#>  purrr      * 0.3.4   2020-04-17 [1] CRAN (R 4.2.0)
#>  recipes    * 1.0.1   2022-07-07 [1] CRAN (R 4.2.0)
#>  rlang        1.0.5   2022-08-31 [1] CRAN (R 4.2.0)
#>  rsample    * 1.1.0   2022-08-08 [1] CRAN (R 4.2.0)
#>  tibble     * 3.1.8   2022-07-22 [1] CRAN (R 4.2.0)
#>  tidymodels * 1.0.0   2022-07-13 [1] CRAN (R 4.2.0)
#>  tune       * 1.0.0   2022-07-07 [1] CRAN (R 4.2.0)
#>  workflows  * 1.0.0   2022-07-05 [1] CRAN (R 4.2.0)
#>  yardstick  * 1.1.0   2022-09-07 [1] CRAN (R 4.2.0)
#> 
#>  [1] /Library/Frameworks/R.framework/Versions/4.2/Resources/library
#> 
#> ─ Python configuration ─────────────────────────────────────────────
#>  python:         /Users/emilhvitfeldthansen/Library/r-miniconda/envs/r-reticulate/bin/python
#>  libpython:      /Users/emilhvitfeldthansen/Library/r-miniconda/envs/r-reticulate/lib/libpython3.7m.dylib
#>  pythonhome:     /Users/emilhvitfeldthansen/Library/r-miniconda/envs/r-reticulate:/Users/emilhvitfeldthansen/Library/r-miniconda/envs/r-reticulate
#>  version:        3.7.11 (default, Jul 27 2021, 07:03:16)  [Clang 10.0.0 ]
#>  numpy:          /Users/emilhvitfeldthansen/Library/r-miniconda/envs/r-reticulate/lib/python3.7/site-packages/numpy
#>  numpy_version:  1.21.6
#> 
#> ────────────────────────────────────────────────────────────────────
```
