---
title: "Tune model parameters"
weight: 4
tags: [rsample, parsnip, tune, dials, workflows, yardstick]
categories: [tuning]
description: | 
  Estimate the best values for hyperparameters that cannot be learned directly during model training.
---






## Introduction {#intro}

Some model parameters cannot be learned directly from a data set during model training; these kinds of parameters are called **hyperparameters**. Some examples of hyperparameters include the number of predictors that are sampled at splits in a tree-based model (we call this `mtry` in tidymodels) or the learning rate in a boosted tree model (we call this `learn_rate`). Instead of learning these kinds of hyperparameters during model training, we can _estimate_ the best values for these values by training many models on resampled data sets and exploring how well all these models perform. This process is called **tuning**.

To use code in this article,  you will need to install the following packages: modeldata, rpart, tidymodels, and vip.


```r
library(tidymodels)  # for the tune package, along with the rest of tidymodels

# Helper packages
library(modeldata)   # for the cells data
library(vip)         # for variable importance plots
```

## The cell image data, revisited {#data}

In our previous [*Evaluate your model with resampling*](/start/resampling/) article, we introduced a data set of images of cells that were labeled by experts as well-segmented (`WS`) or poorly segmented (`PS`). We trained a [random forest model](/start/resampling/#modeling) to predict which images are segmented well vs. poorly, so that a biologist could filter out poorly segmented cell images in their analysis. We used [resampling](/start/resampling/#resampling) to estimate the performance of our model on this data.


```r
data(cells, package = "modeldata")
cells
#> # A tibble: 2,019 x 58
#>   case  class angle_ch_1 area_ch_1 avg_inten_ch_1 avg_inten_ch_2 avg_inten_ch_3
#>   <fct> <fct>      <dbl>     <int>          <dbl>          <dbl>          <dbl>
#> 1 Test  PS        143.         185           15.7           4.95           9.55
#> 2 Train PS        134.         819           31.9         207.            69.9 
#> 3 Train WS        107.         431           28.0         116.            63.9 
#> 4 Train PS         69.2        298           19.5         102.            28.2 
#> 5 Test  PS          2.89       285           24.3         112.            20.5 
#> # … with 2,014 more rows, and 51 more variables: avg_inten_ch_4 <dbl>,
#> #   convex_hull_area_ratio_ch_1 <dbl>, convex_hull_perim_ratio_ch_1 <dbl>,
#> #   diff_inten_density_ch_1 <dbl>, diff_inten_density_ch_3 <dbl>, …
```

## Predicting image segmentation, but better {#why-tune}

Random forest models are a tree-based ensemble method, and typically perform well with [default hyperparameters](https://bradleyboehmke.github.io/HOML/random-forest.html#out-of-the-box-performance). However, the accuracy of some other tree-based models, such as [boosted tree models](https://en.wikipedia.org/wiki/Gradient_boosting#Gradient_tree_boosting) or [decision tree models](https://en.wikipedia.org/wiki/Decision_tree), can be sensitive to the values of hyperparameters. In this article, we will train a **decision tree** model. There are several hyperparameters for decision tree models that can be tuned for better performance. Let's explore:

- the complexity parameter (which we call `cost_complexity` in tidymodels) for the tree, and
- the maximum `tree_depth`.

Tuning these hyperparameters can improve model performance because decision tree models are prone to [overfitting](https://bookdown.org/max/FES/important-concepts.html#overfitting). This happens because single tree models tend to fit the training data _too well_ &mdash; so well, in fact, that they over-learn patterns present in the training data that end up being detrimental when predicting new data. 

We will tune the model hyperparameters to avoid overfitting. Tuning the value of `cost_complexity` helps by [pruning](https://bradleyboehmke.github.io/HOML/DT.html#pruning) back our tree. It adds a cost, or penalty, to error rates of more complex trees; a cost closer to zero decreases the number tree nodes pruned and is more likely to result in an overfit tree. However, a high cost increases the number of tree nodes pruned and can result in the opposite problem&mdash;an underfit tree. Tuning `tree_depth`, on the other hand, helps by [stopping](https://bradleyboehmke.github.io/HOML/DT.html#early-stopping)  our tree from growing after it reaches a certain depth. We want to tune these hyperparameters to find what those two values should be for our model to do the best job predicting image segmentation. 

Before we start the tuning process, we split our data into training and testing sets, just like when we trained the model with one default set of hyperparameters. As [before](/start/resampling/), we can use `strata = class` if we want our training and testing sets to be created using stratified sampling so that both have the same proportion of both kinds of segmentation.


```r
set.seed(123)
cell_split <- initial_split(cells %>% select(-case), 
                            strata = class)
cell_train <- training(cell_split)
cell_test  <- testing(cell_split)
```

We use the training data for tuning the model.

## Tuning hyperparameters {#tuning}

Let’s start with the parsnip package, using a [`decision_tree()`](https://tidymodels.github.io/parsnip/reference/decision_tree.html) model with the [rpart](https://cran.r-project.org/web/packages/rpart/index.html) engine. To tune the decision tree hyperparameters `cost_complexity` and `tree_depth`, we create a model specification that identifies which hyperparameters we plan to tune. 


```r
tune_spec <- 
  decision_tree(
    cost_complexity = tune(),
    tree_depth = tune()
  ) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

tune_spec
#> Decision Tree Model Specification (classification)
#> 
#> Main Arguments:
#>   cost_complexity = tune()
#>   tree_depth = tune()
#> 
#> Computational engine: rpart
```

Think of `tune()` here as a placeholder. After the tuning process, we will select a single numeric value for each of these hyperparameters. For now, we specify our parsnip model object and identify the hyperparameters we will `tune()`.

We can't train this specification on a single data set (such as the entire training set) and learn what the hyperparameter values should be, but we _can_ train many models using resampled data and see which models turn out best. We can create a regular grid of values to try using some convenience functions for each hyperparameter:


```r
tree_grid <- grid_regular(cost_complexity(),
                          tree_depth(),
                          levels = 5)
```

The function [`grid_regular()`](https://tidymodels.github.io/dials/reference/grid_regular.html) is from the [dials](https://tidymodels.github.io/dials/) package. It chooses sensible values to try for each hyperparameter; here, we asked for 5 of each. Since we have two to tune, `grid_regular()` returns 5 `\(\times\)` 5 = 25 different possible tuning combinations to try in a tidy tibble format.


```r
tree_grid
#> # A tibble: 25 x 2
#>    cost_complexity tree_depth
#>              <dbl>      <int>
#>  1    0.0000000001          1
#>  2    0.0000000178          1
#>  3    0.00000316            1
#>  4    0.000562              1
#>  5    0.1                   1
#>  6    0.0000000001          4
#>  7    0.0000000178          4
#>  8    0.00000316            4
#>  9    0.000562              4
#> 10    0.1                   4
#> # … with 15 more rows
```

Here, you can see all 5 values of `cost_complexity` ranging up to 0.1. These values get repeated for each of the 5 values of `tree_depth`:


```r
tree_grid %>% 
  count(tree_depth)
#> # A tibble: 5 x 2
#>   tree_depth     n
#>        <int> <int>
#> 1          1     5
#> 2          4     5
#> 3          8     5
#> 4         11     5
#> 5         15     5
```


Armed with our grid filled with 25 candidate decision tree models, let's create [cross-validation folds](/start/resampling/) for tuning:


```r
set.seed(234)
cell_folds <- vfold_cv(cell_train)
```

Tuning in tidymodels requires a resampled object created with the [rsample](https://tidymodels.github.io/rsample/) package.

## Model tuning with a grid {#tune-grid}

We are ready to tune! Let's use [`tune_grid()`](https://tidymodels.github.io/tune/reference/tune_grid.html) to fit models at all the different values we chose for each tuned hyperparameter. There are several options for building the object for tuning:

+ Tune a model specification along with a recipe or model, or 

+ Tune a [`workflow()`](https://tidymodels.github.io/workflows/) that bundles together a model specification and a recipe or model preprocessor. 

Here we use a `workflow()` with a straightforward formula; if this model required more involved data preprocessing, we could use `add_recipe()` instead of `add_formula()`.


```r
set.seed(345)

tree_wf <- workflow() %>%
  add_model(tune_spec) %>%
  add_formula(class ~ .)

tree_res <- 
  tree_wf %>% 
  tune_grid(
    resamples = cell_folds,
    grid = tree_grid
    )

tree_res
#> # Tuning results
#> # 10-fold cross-validation 
#> # A tibble: 10 x 4
#>    splits             id     .metrics          .notes          
#>    <list>             <chr>  <list>            <list>          
#>  1 <split [1.4K/152]> Fold01 <tibble [50 × 6]> <tibble [0 × 1]>
#>  2 <split [1.4K/152]> Fold02 <tibble [50 × 6]> <tibble [0 × 1]>
#>  3 <split [1.4K/152]> Fold03 <tibble [50 × 6]> <tibble [0 × 1]>
#>  4 <split [1.4K/152]> Fold04 <tibble [50 × 6]> <tibble [0 × 1]>
#>  5 <split [1.4K/152]> Fold05 <tibble [50 × 6]> <tibble [0 × 1]>
#>  6 <split [1.4K/151]> Fold06 <tibble [50 × 6]> <tibble [0 × 1]>
#>  7 <split [1.4K/151]> Fold07 <tibble [50 × 6]> <tibble [0 × 1]>
#>  8 <split [1.4K/151]> Fold08 <tibble [50 × 6]> <tibble [0 × 1]>
#>  9 <split [1.4K/151]> Fold09 <tibble [50 × 6]> <tibble [0 × 1]>
#> 10 <split [1.4K/151]> Fold10 <tibble [50 × 6]> <tibble [0 × 1]>
```

Once we have our tuning results, we can both explore them through visualization and then select the best result. The function `collect_metrics()` gives us a tidy tibble with all the results. We had 25 candidate models and two metrics, `accuracy` and `roc_auc`, and we get a row for each `.metric` and model. 


```r
tree_res %>% 
  collect_metrics()
#> # A tibble: 50 x 8
#>    cost_complexity tree_depth .metric  .estimator  mean     n std_err .config
#>              <dbl>      <int> <chr>    <chr>      <dbl> <int>   <dbl> <chr>  
#>  1    0.0000000001          1 accuracy binary     0.734    10 0.00877 Model01
#>  2    0.0000000001          1 roc_auc  binary     0.772    10 0.00617 Model01
#>  3    0.0000000178          1 accuracy binary     0.734    10 0.00877 Model02
#>  4    0.0000000178          1 roc_auc  binary     0.772    10 0.00617 Model02
#>  5    0.00000316            1 accuracy binary     0.734    10 0.00877 Model03
#>  6    0.00000316            1 roc_auc  binary     0.772    10 0.00617 Model03
#>  7    0.000562              1 accuracy binary     0.734    10 0.00877 Model04
#>  8    0.000562              1 roc_auc  binary     0.772    10 0.00617 Model04
#>  9    0.1                   1 accuracy binary     0.734    10 0.00877 Model05
#> 10    0.1                   1 roc_auc  binary     0.772    10 0.00617 Model05
#> # … with 40 more rows
```

We might get more out of plotting these results:


```r
tree_res %>%
  collect_metrics() %>%
  mutate(tree_depth = factor(tree_depth)) %>%
  ggplot(aes(cost_complexity, mean, color = tree_depth)) +
  geom_line(size = 1.5, alpha = 0.6) +
  geom_point(size = 2) +
  facet_wrap(~ .metric, scales = "free", nrow = 2) +
  scale_x_log10(labels = scales::label_number()) +
  scale_color_viridis_d(option = "plasma", begin = .9, end = 0)
```

<img src="figs/best-tree-1.svg" width="768" />

We can see that our "stubbiest" tree, with a depth of 1, is the worst model according to both metrics and across all candidate values of `cost_complexity`. Our deepest tree, with a depth of 15, did better. However, the best tree seems to be between these values with a tree depth of 4. The [`show_best()`](https://tidymodels.github.io/tune/reference/show_best.html) function shows us the top 5 candidate models by default:


```r
tree_res %>%
  show_best("roc_auc")
#> # A tibble: 5 x 8
#>   cost_complexity tree_depth .metric .estimator  mean     n std_err .config
#>             <dbl>      <int> <chr>   <chr>      <dbl> <int>   <dbl> <chr>  
#> 1    0.0000000001          4 roc_auc binary     0.865    10 0.00965 Model06
#> 2    0.0000000178          4 roc_auc binary     0.865    10 0.00965 Model07
#> 3    0.00000316            4 roc_auc binary     0.865    10 0.00965 Model08
#> 4    0.000562              4 roc_auc binary     0.865    10 0.00965 Model09
#> 5    0.0000000001          8 roc_auc binary     0.859    10 0.0104  Model11
```

We can also use the [`select_best()`](https://tidymodels.github.io/tune/reference/show_best.html) function to pull out the single set of hyperparameter values for our best decision tree model:


```r
best_tree <- tree_res %>%
  select_best("roc_auc")

best_tree
#> # A tibble: 1 x 3
#>   cost_complexity tree_depth .config
#>             <dbl>      <int> <chr>  
#> 1    0.0000000001          4 Model06
```

These are the values for `tree_depth` and `cost_complexity` that maximize AUC in this data set of cell images. 


## Finalizing our model {#final-model}

We can update (or "finalize") our workflow object `tree_wf` with the values from `select_best()`. 


```r
final_wf <- 
  tree_wf %>% 
  finalize_workflow(best_tree)

final_wf
#> ══ Workflow ══════════════════════════════════════════════════════════
#> Preprocessor: Formula
#> Model: decision_tree()
#> 
#> ── Preprocessor ──────────────────────────────────────────────────────
#> class ~ .
#> 
#> ── Model ─────────────────────────────────────────────────────────────
#> Decision Tree Model Specification (classification)
#> 
#> Main Arguments:
#>   cost_complexity = 1e-10
#>   tree_depth = 4
#> 
#> Computational engine: rpart
```

Our tuning is done!

### Exploring results

Let's fit this final model to the training data. What does the decision tree look like?


```r
final_tree <- 
  final_wf %>%
  fit(data = cell_train) 

final_tree
#> ══ Workflow [trained] ════════════════════════════════════════════════
#> Preprocessor: Formula
#> Model: decision_tree()
#> 
#> ── Preprocessor ──────────────────────────────────────────────────────
#> class ~ .
#> 
#> ── Model ─────────────────────────────────────────────────────────────
#> n= 1515 
#> 
#> node), split, n, loss, yval, (yprob)
#>       * denotes terminal node
#> 
#>  1) root 1515 540 PS (0.64356436 0.35643564)  
#>    2) total_inten_ch_2< 47256.5 731  63 PS (0.91381669 0.08618331)  
#>      4) total_inten_ch_2< 37166 585  19 PS (0.96752137 0.03247863) *
#>      5) total_inten_ch_2>=37166 146  44 PS (0.69863014 0.30136986)  
#>       10) avg_inten_ch_1< 99.15056 98  14 PS (0.85714286 0.14285714) *
#>       11) avg_inten_ch_1>=99.15056 48  18 WS (0.37500000 0.62500000)  
#>         22) fiber_align_2_ch_3>=1.47949 20   8 PS (0.60000000 0.40000000) *
#>         23) fiber_align_2_ch_3< 1.47949 28   6 WS (0.21428571 0.78571429) *
#>    3) total_inten_ch_2>=47256.5 784 307 WS (0.39158163 0.60841837)  
#>      6) fiber_width_ch_1< 11.19756 329 137 PS (0.58358663 0.41641337)  
#>       12) avg_inten_ch_1< 194.4183 254  82 PS (0.67716535 0.32283465) *
#>       13) avg_inten_ch_1>=194.4183 75  20 WS (0.26666667 0.73333333)  
#>         26) total_inten_ch_3>=62458.5 23   9 PS (0.60869565 0.39130435) *
#>         27) total_inten_ch_3< 62458.5 52   6 WS (0.11538462 0.88461538) *
#>      7) fiber_width_ch_1>=11.19756 455 115 WS (0.25274725 0.74725275)  
#>       14) shape_p_2_a_ch_1>=1.225676 300  97 WS (0.32333333 0.67666667)  
#>         28) avg_inten_ch_2>=362.0108 55  23 PS (0.58181818 0.41818182) *
#>         29) avg_inten_ch_2< 362.0108 245  65 WS (0.26530612 0.73469388) *
#>       15) shape_p_2_a_ch_1< 1.225676 155  18 WS (0.11612903 0.88387097) *
```

This `final_tree` object has the finalized, fitted model object inside. You may want to extract the model object from the workflow. To do this, you can use the helper function [`pull_workflow_fit()`](https://tidymodels.github.io/workflows/reference/workflow-extractors.html).

For example, perhaps we would also like to understand what variables are important in this final model. We can use the [vip](https://koalaverse.github.io/vip/) package to estimate variable importance. 


```r
library(vip)

final_tree %>% 
  pull_workflow_fit() %>% 
  vip()
```

<img src="figs/vip-1.svg" width="576" />

These are the automated image analysis measurements that are the most important in driving segmentation quality predictions.

### The last fit

Finally, let's return to our test data and estimate the model performance we expect to see with new data. We can use the function [`last_fit()`](https://tidymodels.github.io/tune/reference/last_fit.html) with our finalized model; this function _fits_ the finalized model on the full training data set and _evaluates_ the finalized model on the testing data.


```r
final_fit <- 
  final_wf %>%
  last_fit(cell_split) 

final_fit %>%
  collect_metrics()
#> # A tibble: 2 x 3
#>   .metric  .estimator .estimate
#>   <chr>    <chr>          <dbl>
#> 1 accuracy binary         0.802
#> 2 roc_auc  binary         0.860

final_fit %>%
  collect_predictions() %>% 
  roc_curve(class, .pred_PS) %>% 
  autoplot()
```

<img src="figs/last-fit-1.svg" width="672" />

The performance metrics from the test set indicate that we did not overfit during our tuning procedure.

We leave it to the reader to explore whether you can tune a different decision tree hyperparameter. You can explore the [reference docs](/find/parsnip/#models), or use the `args()` function to see which parsnip object arguments are available:


```r
args(decision_tree)
#> function (mode = "unknown", cost_complexity = NULL, tree_depth = NULL, 
#>     min_n = NULL) 
#> NULL
```

You could tune the other hyperparameter we didn't use here, `min_n`, which sets the minimum `n` to split at any node. This is another early stopping method for decision trees that can help prevent overfitting. Use this [searchable table](/find/parsnip/#model-args) to find the original argument for `min_n` in the rpart package ([hint](https://stat.ethz.ch/R-manual/R-devel/library/rpart/html/rpart.control.html)). See whether you can tune a different combination of hyperparameters and/or values to improve a tree's ability to predict cell segmentation quality.



## Session information


```
#> ─ Session info ───────────────────────────────────────────────────────────────
#>  setting  value                       
#>  version  R version 4.0.2 (2020-06-22)
#>  os       macOS Catalina 10.15.6      
#>  system   x86_64, darwin17.0          
#>  ui       X11                         
#>  language (EN)                        
#>  collate  en_US.UTF-8                 
#>  ctype    en_US.UTF-8                 
#>  tz       America/Denver              
#>  date     2020-07-21                  
#> 
#> ─ Packages ───────────────────────────────────────────────────────────────────
#>  package    * version date       lib source        
#>  broom      * 0.7.0   2020-07-09 [1] CRAN (R 4.0.0)
#>  dials      * 0.0.8   2020-07-08 [1] CRAN (R 4.0.0)
#>  dplyr      * 1.0.0   2020-05-29 [1] CRAN (R 4.0.0)
#>  ggplot2    * 3.3.2   2020-06-19 [1] CRAN (R 4.0.0)
#>  infer      * 0.5.3   2020-07-14 [1] CRAN (R 4.0.2)
#>  modeldata  * 0.0.2   2020-06-22 [1] CRAN (R 4.0.2)
#>  parsnip    * 0.1.2   2020-07-03 [1] CRAN (R 4.0.1)
#>  purrr      * 0.3.4   2020-04-17 [1] CRAN (R 4.0.0)
#>  recipes    * 0.1.13  2020-06-23 [1] CRAN (R 4.0.0)
#>  rlang        0.4.7   2020-07-09 [1] CRAN (R 4.0.2)
#>  rpart      * 4.1-15  2019-04-12 [1] CRAN (R 4.0.2)
#>  rsample    * 0.0.7   2020-06-04 [1] CRAN (R 4.0.0)
#>  tibble     * 3.0.3   2020-07-10 [1] CRAN (R 4.0.2)
#>  tidymodels * 0.1.1   2020-07-14 [1] CRAN (R 4.0.2)
#>  tune       * 0.1.1   2020-07-08 [1] CRAN (R 4.0.0)
#>  vip        * 0.2.2   2020-04-06 [1] CRAN (R 4.0.0)
#>  workflows  * 0.1.2   2020-07-07 [1] CRAN (R 4.0.0)
#>  yardstick  * 0.0.7   2020-07-13 [1] CRAN (R 4.0.2)
#> 
#> [1] /Library/Frameworks/R.framework/Versions/4.0/Resources/library
```
