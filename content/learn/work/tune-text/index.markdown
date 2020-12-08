---
title: "Tuning text models"
tags: [rsample, recipes, textrecipes, parsnip, tune, yardstick]
categories: [model tuning]
type: learn-subsection
weight: 4
description: | 
  Prepare text data for predictive modeling and tune with both grid and iterative search.
---





## Introduction

To use the code in this article, you will need to install the following packages: stopwords, textfeatures, textrecipes, and tidymodels.

This article demonstrates an advanced example for training and tuning models for text data. Text data must be processed and transformed to a numeric representation to be ready for computation in modeling; in tidymodels, we use a recipe for this preprocessing. This article also shows how to extract information from each model fit during tuning to use later on.


## Text as data

The text data we'll use in this article are from Amazon: 

> This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plaintext review.

This article uses a small subset of the total reviews [available at the original source](https://snap.stanford.edu/data/web-FineFoods.html). We sampled a single review from 5,000 random products and allocated 80% of these data to the training set, with the remaining 1,000 reviews held out for the test set.   

There is a column for the product, a column for the text of the review, and a factor column for the outcome variable. The outcome is whether the reviewer gave the product a five-star rating or not.


```r
library(tidymodels)

data("small_fine_foods")
training_data
#> # A tibble: 4,000 x 3
#>    product    review                                                       score
#>    <chr>      <chr>                                                        <fct>
#>  1 B000J0LSBG "this stuff is  not stuffing  its  not good at all  save yo… other
#>  2 B000EYLDYE "I absolutely LOVE this dried fruit.  LOVE IT.  Whenever I … great
#>  3 B0026LIO9A "GREAT DEAL, CONVENIENT TOO.  Much cheaper than WalMart and… great
#>  4 B00473P8SK "Great flavor, we go through a ton of this sauce! I discove… great
#>  5 B001SAWTNM "This is excellent salsa/hot sauce, but you can get it for … great
#>  6 B000FAG90U "Again, this is the best dogfood out there.  One suggestion… great
#>  7 B006BXTCEK "The box I received was filled with teas, hot chocolates, a… other
#>  8 B002GWH5OY "This is delicious coffee which compares favorably with muc… great
#>  9 B003R0MFYY "Don't let these little tiny cans fool you.  They pack a lo… great
#> 10 B001EO5ZXI "One of the nicest, smoothest cup of chai I've made. Nice m… great
#> # … with 3,990 more rows
```

Our modeling goal is to create modeling features from the text of the reviews to predict whether the review was five-star or not.

## Inputs for the search

Text, perhaps more so than tabular data we often deal with, must be heavily processed to be used as predictor data for modeling. There are multiple ways to process and prepare text for modeling; let's add several steps together to create different kinds of features:

* Create an initial set of count-based features, such as the number of words, spaces, lower- or uppercase characters, URLs, and so on; we can use the [textfeatures](https://github.com/mkearney/textfeatures) package for this.

* [Tokenize](https://smltar.com/tokenization.html) the text (i.e. break the text into smaller components such as words).

* Remove stop words such as "the", "an", "of", etc.

* [Stem](https://smltar.com/stemming.html) tokens to a common root where possible.

* Convert tokens to dummy variables via a [signed, binary hash function](https://bookdown.org/max/FES/encoding-predictors-with-many-categories.html).

* Optionally transform non-token features (the count-based features like number of lowercase characters) to a more symmetric state using a [Yeo-Johnson transformation](https://bookdown.org/max/FES/numeric-one-to-one.html).

* Remove predictors with a single distinct value.

* Center and scale all predictors. 


{{% note %}}  We will end up with two kinds of features:

- dummy/indicator variables for the count-based features like number of digits or punctuation characters 
- hash features for the tokens like "salsa" or "delicious". {{%/ note %}}

Some of these preprocessing steps (such as stemming) may or may not be good ideas but a full discussion of their effects is beyond the scope of this article. In this preprocessing approach, the main tuning parameter is the number of hashing features to use. 

Before we start building our preprocessing recipe, we need some helper objects. For example, for the Yeo-Johnson transformation, we need to know the set of count-based text features: 


```r
library(textfeatures)

basics <- names(textfeatures:::count_functions)
head(basics)
#> [1] "n_words"    "n_uq_words" "n_charS"    "n_uq_charS" "n_digits"  
#> [6] "n_hashtags"
```

Also, the implementation of feature hashes does not produce the binary values we need. This small function will help convert the scores to values of -1, 0, or 1:


```r
binary_hash <- function(x) {
  x <- ifelse(x < 0, -1, x)
  x <- ifelse(x > 0,  1, x)
  x
}
```

Now, let's put this all together in one recipe:


```r
library(textrecipes)

pre_proc <-
  recipe(score ~ product + review, data = training_data) %>%
  # Do not use the product ID as a predictor
  update_role(product, new_role = "id") %>%
  # Make a copy of the raw text
  step_mutate(review_raw = review) %>%
  # Compute the initial features. This removes the `review_raw` column
  step_textfeature(review_raw) %>%
  # Make the feature names shorter
  step_rename_at(
    starts_with("textfeature_"),
    fn = ~ gsub("textfeature_review_raw_", "", .)
  ) %>%
  step_tokenize(review)  %>%
  step_stopwords(review) %>%
  step_stem(review) %>%
  # Here is where the tuning parameter is declared
  step_texthash(review, signed = TRUE, num_terms = tune()) %>%
  # Simplify these names
  step_rename_at(starts_with("review_hash"), fn = ~ gsub("review_", "", .)) %>%
  # Convert the features from counts to values of -1, 0, or 1
  step_mutate_at(starts_with("hash"), fn = binary_hash) %>%
  # Transform the initial feature set
  step_YeoJohnson(one_of(!!basics)) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())
```

{{% warning %}} Note that, when objects from the global environment are used, they are injected into the step objects via `!!`. For some parallel processing technologies, these objects may not be found by the worker processes. {{%/ warning %}}

The preprocessing recipe is long and complex (often typical for working with text data) but the model we'll use is more straightforward. Let's stick with a regularized logistic regression model: 


```r
lr_mod <-
  logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")
```

There are three tuning parameters for this data analysis:

- `num_terms`, the number of feature hash variables to create
- `penalty`, the amount of regularization for the model
- `mixture`, the proportion of L1 regularization

## Resampling

There are enough data here so that 10-fold resampling would hold out 400 reviews at a time to estimate performance. Performance estimates using this many observations have sufficiently low noise to measure and tune models.  


```r
set.seed(8935)
folds <- vfold_cv(training_data)
folds
#> #  10-fold cross-validation 
#> # A tibble: 10 x 2
#>    splits             id    
#>    <list>             <chr> 
#>  1 <split [3.6K/400]> Fold01
#>  2 <split [3.6K/400]> Fold02
#>  3 <split [3.6K/400]> Fold03
#>  4 <split [3.6K/400]> Fold04
#>  5 <split [3.6K/400]> Fold05
#>  6 <split [3.6K/400]> Fold06
#>  7 <split [3.6K/400]> Fold07
#>  8 <split [3.6K/400]> Fold08
#>  9 <split [3.6K/400]> Fold09
#> 10 <split [3.6K/400]> Fold10
```

## Grid search

Let's begin our tuning with [grid search](https://www.tidymodels.org/learn/work/tune-svm/) and a regular grid. For glmnet models, evaluating penalty values is fairly cheap because of the use of the ["submodel-trick"](https://tune.tidymodels.org/articles/extras/optimizations.html#sub-model-speed-ups-1). The grid will use 20 penalty values, 5 mixture values, and 3 values for the number of hash features.  


```r
five_star_grid <- 
  crossing(
    penalty = 10^seq(-3, 0, length = 20),
    mixture = c(0.01, 0.25, 0.50, 0.75, 1),
    num_terms = 2^c(8, 10, 12)
  )
five_star_grid
#> # A tibble: 300 x 3
#>    penalty mixture num_terms
#>      <dbl>   <dbl>     <dbl>
#>  1   0.001    0.01       256
#>  2   0.001    0.01      1024
#>  3   0.001    0.01      4096
#>  4   0.001    0.25       256
#>  5   0.001    0.25      1024
#>  6   0.001    0.25      4096
#>  7   0.001    0.5        256
#>  8   0.001    0.5       1024
#>  9   0.001    0.5       4096
#> 10   0.001    0.75       256
#> # … with 290 more rows
```

Note that, for each resample, the (computationally expensive) text preprocessing recipe is only prepped 6 times. This increases the efficiency of the analysis by avoiding redundant work. 

Let's save information on the number of predictors by penalty value for each glmnet model. This can help us understand how many features were used across the penalty values. Use an extraction function to do this:


```r
glmnet_vars <- function(x) {
  # `x` will be a workflow object
  mod <- extract_model(x)
  # `df` is the number of model terms for each penalty value
  tibble(penalty = mod$lambda, num_vars = mod$df)
}

ctrl <- control_grid(extract = glmnet_vars, verbose = TRUE)
```

Finally, let's run the grid search:


```r
roc_scores <- metric_set(roc_auc)

set.seed(1559)
five_star_glmnet <- 
  tune_grid(
    lr_mod, 
    pre_proc, 
    resamples = folds, 
    grid = five_star_grid, 
    metrics = roc_scores, 
    control = ctrl
  )

five_star_glmnet
#> # Tuning results
#> # 10-fold cross-validation 
#> # A tibble: 10 x 5
#>    splits             id     .metrics          .notes          .extracts        
#>    <list>             <chr>  <list>            <list>          <list>           
#>  1 <split [3.6K/400]> Fold01 <tibble [300 × 7… <tibble [0 × 1… <tibble [300 × 5…
#>  2 <split [3.6K/400]> Fold02 <tibble [300 × 7… <tibble [0 × 1… <tibble [300 × 5…
#>  3 <split [3.6K/400]> Fold03 <tibble [300 × 7… <tibble [0 × 1… <tibble [300 × 5…
#>  4 <split [3.6K/400]> Fold04 <tibble [300 × 7… <tibble [0 × 1… <tibble [300 × 5…
#>  5 <split [3.6K/400]> Fold05 <tibble [300 × 7… <tibble [0 × 1… <tibble [300 × 5…
#>  6 <split [3.6K/400]> Fold06 <tibble [300 × 7… <tibble [0 × 1… <tibble [300 × 5…
#>  7 <split [3.6K/400]> Fold07 <tibble [300 × 7… <tibble [0 × 1… <tibble [300 × 5…
#>  8 <split [3.6K/400]> Fold08 <tibble [300 × 7… <tibble [0 × 1… <tibble [300 × 5…
#>  9 <split [3.6K/400]> Fold09 <tibble [300 × 7… <tibble [0 × 1… <tibble [300 × 5…
#> 10 <split [3.6K/400]> Fold10 <tibble [300 × 7… <tibble [0 × 1… <tibble [300 × 5…
```

This took a while to complete! What do the results look like? Let's get the resampling estimates of the area under the ROC curve for each tuning parameter:


```r
grid_roc <- 
  collect_metrics(five_star_glmnet) %>% 
  arrange(desc(mean))
grid_roc
#> # A tibble: 300 x 9
#>    penalty mixture num_terms .metric .estimator  mean     n std_err .config     
#>      <dbl>   <dbl>     <dbl> <chr>   <chr>      <dbl> <int>   <dbl> <chr>       
#>  1 0.695      0.01      4096 roc_auc binary     0.818    10 0.00739 Preprocesso…
#>  2 0.483      0.01      4096 roc_auc binary     0.818    10 0.00776 Preprocesso…
#>  3 0.0379     0.25      4096 roc_auc binary     0.816    10 0.00753 Preprocesso…
#>  4 0.336      0.01      4096 roc_auc binary     0.815    10 0.00822 Preprocesso…
#>  5 0.0183     0.5       4096 roc_auc binary     0.814    10 0.00776 Preprocesso…
#>  6 0.0127     0.75      4096 roc_auc binary     0.814    10 0.00781 Preprocesso…
#>  7 0.00886    1         4096 roc_auc binary     0.813    10 0.00794 Preprocesso…
#>  8 0.0264     0.25      4096 roc_auc binary     0.813    10 0.00790 Preprocesso…
#>  9 1          0.01      4096 roc_auc binary     0.812    10 0.00764 Preprocesso…
#> 10 0.234      0.01      4096 roc_auc binary     0.811    10 0.00838 Preprocesso…
#> # … with 290 more rows
```

The best results have a fairly high penalty value and focus on the ridge penalty (i.e. no feature selection via the lasso's L1 penalty). The best solutions also use the largest number of hashing features. 

What is the relationship between performance and the tuning parameters? 


```r
autoplot(five_star_glmnet, metric = "roc_auc")
```

<img src="figs/grid-plot-1.svg" width="960" />

- We can definitely see that performance improves with the number of features included. In this article, we've used a small sample of the overall data set available. When more data are used, an even larger feature set is optimal. 

- The profiles with larger mixture values (greater than 0.01) have steep drop-offs in performance. What's that about? Those are cases where the lasso penalty is removing too many (and perhaps all) features from the model.  
- The panel with at least 4096 features shows that there are several parameter combinations that have about the same performance; there isn't much difference between the best performance for the different mixture values. A case could be made that we should choose a _larger_ mixture value and a _smaller_ penalty to select a simpler model that contains fewer predictors. 

- If more experimentation were conducted, a larger set of features (more than 4096) should also be considered.  

We'll come back to the extracted glmnet components at the end of this article. 

## Directed search

What if we had started with Bayesian optimization? Would a good set of conditions have been found more efficiently? 

Let's pretend that we haven't seen the grid search results. We'll initialize the Gaussian process model with five tuning parameter combinations chosen with a space-filling design. 

It might be good to use a custom `dials` object for the number of hash terms. The default object, `num_terms()`, uses a linear range and tries to set the upper bound of the parameter using the data. Instead, let's create a parameter set, change the scale to be `log2`, and define the same range as was used in grid search. 


```r
hash_range <- num_terms(c(8, 12), trans = log2_trans())
hash_range
#> # Model Terms (quantitative)
#> Transformer:  log-2 
#> Range (transformed scale): [8, 12]
```

To use this, we have to merge the recipe and `parsnip` model object into a workflow:


```r
five_star_wflow <-
  workflow() %>%
  add_recipe(pre_proc) %>%
  add_model(lr_mod)
```

Then we can extract and manipulate the corresponding parameter set:


```r
five_star_set <-
  five_star_wflow %>%
  parameters() %>%
  update(
    num_terms = hash_range, 
    penalty = penalty(c(-3, 0)),
    mixture = mixture(c(0.05, 1.00))
  )
```

This is passed to the search function via the `param_info` argument. 

The initial rounds of search can be biased more towards exploration of the parameter space (as opposed to staying near the current best results). If expected improvement is used as the acquisition function, the trade-off value can be slowly moved from exploration to exploitation over iterations (see the tune vignette on [acquisition functions](https://tune.tidymodels.org/articles/acquisition_functions.html) for more details). The tune package has a built-in function called `expo_decay()` that can help accomplish this:


```r
trade_off_decay <- function(iter) {
  expo_decay(iter, start_val = .01, limit_val = 0, slope = 1/4)
}
```

Using these values, let's run the search:


```r
set.seed(12)
five_star_search <-
  tune_bayes(
    five_star_wflow, 
    resamples = folds,
    param_info = five_star_set,
    initial = 5,
    iter = 30,
    metrics = roc_scores,
    objective = exp_improve(trade_off_decay),
    control = control_bayes(verbose = TRUE)
  )
#> 
#> >  Generating a set of 5 initial parameter results
#> ✓ Initialization complete
#> 
#> Optimizing roc_auc using the expected improvement with variable trade-off
#> values.
#> 
#> ── Iteration 1 ───────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.7559 (@iter 0)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 0.01
#> i penalty=0.00251, mixture=0.0528, num_terms=260
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7475 (+/-0.0102)
#> 
#> ── Iteration 2 ───────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.7559 (@iter 0)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 0.007788
#> i penalty=0.00389, mixture=0.616, num_terms=489
#> i Estimating performance
#> ✓ Estimating performance
#> ♥ Newest results:	roc_auc=0.7753 (+/-0.009)
#> 
#> ── Iteration 3 ───────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.7753 (@iter 2)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 0.006065
#> i penalty=0.00101, mixture=0.463, num_terms=517
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7456 (+/-0.0107)
#> 
#> ── Iteration 4 ───────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.7753 (@iter 2)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 0.004724
#> i penalty=0.00112, mixture=0.807, num_terms=4072
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7703 (+/-0.00716)
#> 
#> ── Iteration 5 ───────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.7753 (@iter 2)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 0.003679
#> i penalty=0.00343, mixture=0.523, num_terms=4038
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7751 (+/-0.00694)
#> 
#> ── Iteration 6 ───────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.7753 (@iter 2)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 0.002865
#> i penalty=0.00306, mixture=0.989, num_terms=1251
#> i Estimating performance
#> ✓ Estimating performance
#> ♥ Newest results:	roc_auc=0.7859 (+/-0.00933)
#> 
#> ── Iteration 7 ───────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.7859 (@iter 6)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 0.002231
#> i penalty=0.0174, mixture=0.991, num_terms=2349
#> i Estimating performance
#> ✓ Estimating performance
#> ♥ Newest results:	roc_auc=0.7981 (+/-0.0078)
#> 
#> ── Iteration 8 ───────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.7981 (@iter 7)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 0.001738
#> i penalty=0.011, mixture=0.999, num_terms=1832
#> i Estimating performance
#> ✓ Estimating performance
#> ♥ Newest results:	roc_auc=0.8022 (+/-0.00809)
#> 
#> ── Iteration 9 ───────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8022 (@iter 8)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 0.001353
#> i penalty=0.00632, mixture=0.943, num_terms=2806
#> i Estimating performance
#> ✓ Estimating performance
#> ♥ Newest results:	roc_auc=0.8035 (+/-0.0047)
#> 
#> ── Iteration 10 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8035 (@iter 9)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 0.001054
#> i penalty=0.00701, mixture=0.969, num_terms=3717
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8034 (+/-0.0061)
#> 
#> ── Iteration 11 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8035 (@iter 9)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 0.0008208
#> i penalty=0.00554, mixture=0.99, num_terms=3414
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8028 (+/-0.00628)
#> 
#> ── Iteration 12 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8035 (@iter 9)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 0.0006393
#> i penalty=0.0137, mixture=0.987, num_terms=3233
#> i Estimating performance
#> ✓ Estimating performance
#> ♥ Newest results:	roc_auc=0.8162 (+/-0.00823)
#> 
#> ── Iteration 13 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8162 (@iter 12)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 0.0004979
#> i penalty=0.0136, mixture=0.967, num_terms=290
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7445 (+/-0.00702)
#> 
#> ── Iteration 14 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8162 (@iter 12)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 0.0003877
#> i penalty=0.0143, mixture=0.921, num_terms=4085
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.814 (+/-0.00775)
#> 
#> ── Iteration 15 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8162 (@iter 12)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 0.000302
#> i penalty=0.013, mixture=0.973, num_terms=3974
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8099 (+/-0.00471)
#> 
#> ── Iteration 16 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8162 (@iter 12)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 0.0002352
#> i penalty=0.0148, mixture=0.0565, num_terms=3023
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7757 (+/-0.00876)
#> 
#> ── Iteration 17 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8162 (@iter 12)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 0.0001832
#> i penalty=0.0139, mixture=0.992, num_terms=3427
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8021 (+/-0.00829)
#> 
#> ── Iteration 18 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8162 (@iter 12)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 0.0001426
#> i penalty=0.152, mixture=0.0898, num_terms=3286
#> i Estimating performance
#> ✓ Estimating performance
#> ♥ Newest results:	roc_auc=0.8206 (+/-0.00922)
#> 
#> ── Iteration 19 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8206 (@iter 18)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 0.0001111
#> i penalty=0.0978, mixture=0.121, num_terms=3755
#> i Estimating performance
#> ✓ Estimating performance
#> ♥ Newest results:	roc_auc=0.8206 (+/-0.00836)
#> 
#> ── Iteration 20 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8206 (@iter 19)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 8.652e-05
#> i penalty=0.575, mixture=0.156, num_terms=366
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.5
#> 
#> ── Iteration 21 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8206 (@iter 19)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 6.738e-05
#> i penalty=0.0705, mixture=0.108, num_terms=2883
#> i Estimating performance
#> ✓ Estimating performance
#> ♥ Newest results:	roc_auc=0.8314 (+/-0.00731)
#> 
#> ── Iteration 22 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8314 (@iter 21)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 5.248e-05
#> i penalty=0.00145, mixture=0.654, num_terms=2857
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7585 (+/-0.0102)
#> 
#> ── Iteration 23 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8314 (@iter 21)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 4.087e-05
#> i penalty=0.0633, mixture=0.26, num_terms=297
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7461 (+/-0.00837)
#> 
#> ── Iteration 24 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8314 (@iter 21)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 3.183e-05
#> i penalty=0.00393, mixture=0.421, num_terms=257
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7489 (+/-0.00842)
#> 
#> ── Iteration 25 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8314 (@iter 21)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 2.479e-05
#> i penalty=0.0129, mixture=0.16, num_terms=2730
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7876 (+/-0.00503)
#> 
#> ── Iteration 26 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8314 (@iter 21)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 1.93e-05
#> i penalty=0.0659, mixture=0.0836, num_terms=3179
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8094 (+/-0.00417)
#> 
#> ── Iteration 27 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8314 (@iter 21)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 1.503e-05
#> i penalty=0.181, mixture=0.13, num_terms=2383
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7755 (+/-0.0098)
#> 
#> ── Iteration 28 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8314 (@iter 21)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 1.171e-05
#> i penalty=0.257, mixture=0.622, num_terms=3937
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.5
#> 
#> ── Iteration 29 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8314 (@iter 21)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 9.119e-06
#> i penalty=0.0122, mixture=0.103, num_terms=744
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7777 (+/-0.00696)
#> 
#> ── Iteration 30 ──────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8314 (@iter 21)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Trade-off value: 7.102e-06
#> i penalty=0.00135, mixture=0.0737, num_terms=3215
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.726 (+/-0.0066)

five_star_search
#> # Tuning results
#> # 10-fold cross-validation 
#> # A tibble: 310 x 5
#>    splits             id     .metrics         .notes           .iter
#>    <list>             <chr>  <list>           <list>           <int>
#>  1 <split [3.6K/400]> Fold01 <tibble [5 × 7]> <tibble [0 × 1]>     0
#>  2 <split [3.6K/400]> Fold02 <tibble [5 × 7]> <tibble [0 × 1]>     0
#>  3 <split [3.6K/400]> Fold03 <tibble [5 × 7]> <tibble [0 × 1]>     0
#>  4 <split [3.6K/400]> Fold04 <tibble [5 × 7]> <tibble [0 × 1]>     0
#>  5 <split [3.6K/400]> Fold05 <tibble [5 × 7]> <tibble [0 × 1]>     0
#>  6 <split [3.6K/400]> Fold06 <tibble [5 × 7]> <tibble [0 × 1]>     0
#>  7 <split [3.6K/400]> Fold07 <tibble [5 × 7]> <tibble [0 × 1]>     0
#>  8 <split [3.6K/400]> Fold08 <tibble [5 × 7]> <tibble [0 × 1]>     0
#>  9 <split [3.6K/400]> Fold09 <tibble [5 × 7]> <tibble [0 × 1]>     0
#> 10 <split [3.6K/400]> Fold10 <tibble [5 × 7]> <tibble [0 × 1]>     0
#> # … with 300 more rows
```

These results show some improvement over the initial set. One issue is that so many settings are sub-optimal (as shown in the plot above for grid search) so there are poor results periodically. There are regions where the penalty parameter becomes too large and all of the predictors are removed from the model. These regions are also dependent on the number of terms. There is a fairly narrow ridge (sorry, pun intended!) where good performance can be achieved. Using more iterations would probably result in the search finding better results. 
Let's look at a plot of model performance versus the search iterations:


```r
autoplot(five_star_search, type = "performance")
```

<img src="figs/iter-plot-1.svg" width="672" />

{{% note %}} What would we do if we knew about the grid search results and wanted to try directed, iterative search? We would restrict the range for the number of hash features to be larger (especially with more data). We might also restrict the penalty and mixture parameters to have a lower upper bound. {{%/ note %}}

## Extracted results

Let's return to the grid search results and examine the results of our `extract` function. For each _fitted model_, a tibble was saved that contains the relationship between the number of predictors and the penalty value. Let's look at these results for the best model:


```r
params <- select_best(five_star_glmnet, metric = "roc_auc")
params
#> # A tibble: 1 x 4
#>   penalty mixture num_terms .config               
#>     <dbl>   <dbl>     <dbl> <chr>                 
#> 1   0.695    0.01      4096 Preprocessor3_Model019
```

Recall that we saved the glmnet results in a tibble. The column `five_star_glmnet$.extracts` is a list of tibbles. As an example, the first element of the list is:


```r
five_star_glmnet$.extracts[[1]]
#> # A tibble: 300 x 5
#>    num_terms penalty mixture .extracts          .config               
#>        <dbl>   <dbl>   <dbl> <list>             <chr>                 
#>  1       256       1    0.01 <tibble [100 × 2]> Preprocessor1_Model001
#>  2       256       1    0.01 <tibble [100 × 2]> Preprocessor1_Model002
#>  3       256       1    0.01 <tibble [100 × 2]> Preprocessor1_Model003
#>  4       256       1    0.01 <tibble [100 × 2]> Preprocessor1_Model004
#>  5       256       1    0.01 <tibble [100 × 2]> Preprocessor1_Model005
#>  6       256       1    0.01 <tibble [100 × 2]> Preprocessor1_Model006
#>  7       256       1    0.01 <tibble [100 × 2]> Preprocessor1_Model007
#>  8       256       1    0.01 <tibble [100 × 2]> Preprocessor1_Model008
#>  9       256       1    0.01 <tibble [100 × 2]> Preprocessor1_Model009
#> 10       256       1    0.01 <tibble [100 × 2]> Preprocessor1_Model010
#> # … with 290 more rows
```

More nested tibbles! Let's `unnest()` the `five_star_glmnet$.extracts` column:


```r
library(tidyr)
extracted <- 
  five_star_glmnet %>% 
  dplyr::select(id, .extracts) %>% 
  unnest(cols = .extracts)
extracted
#> # A tibble: 3,000 x 6
#>    id     num_terms penalty mixture .extracts          .config               
#>    <chr>      <dbl>   <dbl>   <dbl> <list>             <chr>                 
#>  1 Fold01       256       1    0.01 <tibble [100 × 2]> Preprocessor1_Model001
#>  2 Fold01       256       1    0.01 <tibble [100 × 2]> Preprocessor1_Model002
#>  3 Fold01       256       1    0.01 <tibble [100 × 2]> Preprocessor1_Model003
#>  4 Fold01       256       1    0.01 <tibble [100 × 2]> Preprocessor1_Model004
#>  5 Fold01       256       1    0.01 <tibble [100 × 2]> Preprocessor1_Model005
#>  6 Fold01       256       1    0.01 <tibble [100 × 2]> Preprocessor1_Model006
#>  7 Fold01       256       1    0.01 <tibble [100 × 2]> Preprocessor1_Model007
#>  8 Fold01       256       1    0.01 <tibble [100 × 2]> Preprocessor1_Model008
#>  9 Fold01       256       1    0.01 <tibble [100 × 2]> Preprocessor1_Model009
#> 10 Fold01       256       1    0.01 <tibble [100 × 2]> Preprocessor1_Model010
#> # … with 2,990 more rows
```

One thing to realize here is that `tune_grid()` [may not fit all of the models](https://tune.tidymodels.org/articles/extras/optimizations.html) that are evaluated. In this case, for each value of `mixture` and `num_terms`, the model is fit over _all_ penalty values (this is a feature of this particular model and is not generally true for other engines). To select the best parameter set, we can exclude the `penalty` column in `extracted`:



```r
extracted <- 
  extracted %>% 
  dplyr::select(-penalty) %>% 
  inner_join(params, by = c("num_terms", "mixture")) %>% 
  # Now remove it from the final results
  dplyr::select(-penalty)
extracted
#> # A tibble: 200 x 6
#>    id     num_terms mixture .extracts       .config.x          .config.y        
#>    <chr>      <dbl>   <dbl> <list>          <chr>              <chr>            
#>  1 Fold01      4096    0.01 <tibble [100 ×… Preprocessor3_Mod… Preprocessor3_Mo…
#>  2 Fold01      4096    0.01 <tibble [100 ×… Preprocessor3_Mod… Preprocessor3_Mo…
#>  3 Fold01      4096    0.01 <tibble [100 ×… Preprocessor3_Mod… Preprocessor3_Mo…
#>  4 Fold01      4096    0.01 <tibble [100 ×… Preprocessor3_Mod… Preprocessor3_Mo…
#>  5 Fold01      4096    0.01 <tibble [100 ×… Preprocessor3_Mod… Preprocessor3_Mo…
#>  6 Fold01      4096    0.01 <tibble [100 ×… Preprocessor3_Mod… Preprocessor3_Mo…
#>  7 Fold01      4096    0.01 <tibble [100 ×… Preprocessor3_Mod… Preprocessor3_Mo…
#>  8 Fold01      4096    0.01 <tibble [100 ×… Preprocessor3_Mod… Preprocessor3_Mo…
#>  9 Fold01      4096    0.01 <tibble [100 ×… Preprocessor3_Mod… Preprocessor3_Mo…
#> 10 Fold01      4096    0.01 <tibble [100 ×… Preprocessor3_Mod… Preprocessor3_Mo…
#> # … with 190 more rows
```

Now we can get at the results that we want using another `unnest()`:


```r
extracted <- 
  extracted %>% 
  unnest(col = .extracts) # <- these contain a `penalty` column
extracted
#> # A tibble: 20,000 x 7
#>    id     num_terms mixture penalty num_vars .config.x         .config.y        
#>    <chr>      <dbl>   <dbl>   <dbl>    <int> <chr>             <chr>            
#>  1 Fold01      4096    0.01    9.01        0 Preprocessor3_Mo… Preprocessor3_Mo…
#>  2 Fold01      4096    0.01    8.60        1 Preprocessor3_Mo… Preprocessor3_Mo…
#>  3 Fold01      4096    0.01    8.21        2 Preprocessor3_Mo… Preprocessor3_Mo…
#>  4 Fold01      4096    0.01    7.84        2 Preprocessor3_Mo… Preprocessor3_Mo…
#>  5 Fold01      4096    0.01    7.48        3 Preprocessor3_Mo… Preprocessor3_Mo…
#>  6 Fold01      4096    0.01    7.14        3 Preprocessor3_Mo… Preprocessor3_Mo…
#>  7 Fold01      4096    0.01    6.82        4 Preprocessor3_Mo… Preprocessor3_Mo…
#>  8 Fold01      4096    0.01    6.51        6 Preprocessor3_Mo… Preprocessor3_Mo…
#>  9 Fold01      4096    0.01    6.21        7 Preprocessor3_Mo… Preprocessor3_Mo…
#> 10 Fold01      4096    0.01    5.93        7 Preprocessor3_Mo… Preprocessor3_Mo…
#> # … with 19,990 more rows
```

Let's look at a plot of these results (per resample):


```r
ggplot(extracted, aes(x = penalty, y = num_vars)) + 
  geom_line(aes(group = id, col = id), alpha = .5) + 
  ylab("Number of retained predictors") + 
  scale_x_log10()  + 
  ggtitle(paste("mixture = ", params$mixture, "and", params$num_terms, "features")) + 
  theme(legend.position = "none")
```

<img src="figs/var-plot-1.svg" width="672" />

These results might help guide the choice of the `penalty` range if more optimization was conducted. 

## Session information


```
#> ─ Session info ───────────────────────────────────────────────────────────────
#>  setting  value                       
#>  version  R version 4.0.3 (2020-10-10)
#>  os       macOS Mojave 10.14.6        
#>  system   x86_64, darwin17.0          
#>  ui       X11                         
#>  language (EN)                        
#>  collate  en_US.UTF-8                 
#>  ctype    en_US.UTF-8                 
#>  tz       America/Denver              
#>  date     2020-12-07                  
#> 
#> ─ Packages ───────────────────────────────────────────────────────────────────
#>  package      * version date       lib source        
#>  broom        * 0.7.2   2020-10-20 [1] CRAN (R 4.0.2)
#>  dials        * 0.0.9   2020-09-16 [1] CRAN (R 4.0.2)
#>  dplyr        * 1.0.2   2020-08-18 [1] CRAN (R 4.0.2)
#>  ggplot2      * 3.3.2   2020-06-19 [1] CRAN (R 4.0.0)
#>  infer        * 0.5.3   2020-07-14 [1] CRAN (R 4.0.0)
#>  parsnip      * 0.1.4   2020-10-27 [1] CRAN (R 4.0.2)
#>  purrr        * 0.3.4   2020-04-17 [1] CRAN (R 4.0.0)
#>  recipes      * 0.1.15  2020-11-11 [1] CRAN (R 4.0.2)
#>  rlang        * 0.4.9   2020-11-26 [1] CRAN (R 4.0.2)
#>  rsample      * 0.0.8   2020-09-23 [1] CRAN (R 4.0.2)
#>  stopwords    * 2.0     2020-04-14 [1] CRAN (R 4.0.0)
#>  textfeatures * 0.3.3   2019-09-03 [1] CRAN (R 4.0.2)
#>  textrecipes  * 0.4.0   2020-11-12 [1] CRAN (R 4.0.2)
#>  tibble       * 3.0.4   2020-10-12 [1] CRAN (R 4.0.2)
#>  tidymodels   * 0.1.2   2020-11-22 [1] CRAN (R 4.0.2)
#>  tune         * 0.1.2   2020-11-17 [1] CRAN (R 4.0.3)
#>  workflows    * 0.2.1   2020-10-08 [1] CRAN (R 4.0.2)
#>  yardstick    * 0.0.7   2020-07-13 [1] CRAN (R 4.0.2)
#> 
#> [1] /Library/Frameworks/R.framework/Versions/4.0/Resources/library
```
