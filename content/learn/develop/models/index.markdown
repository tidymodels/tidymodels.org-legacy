---
title: "How to build a parsnip model"
tags: [parsnip]
categories: []
type: learn-subsection
weight: 2
description: | 
  Create a parsnip model function from an existing model implementation.
---





## Introduction

To use the code in this article, you will need to install the following packages: mda, modeldata, and tidymodels.

The parsnip package constructs models and predictions by representing those actions in expressions. There are a few reasons for this:

 * It eliminates a lot of duplicate code.
 * Since the expressions are not evaluated until fitting, it eliminates many package dependencies.

A parsnip model function is itself very general. For example, the `logistic_reg()` function itself doesn't have any model code within it. Instead, each model function is associated with one or more computational _engines_. These might be different R packages or some function in another language (that can be evaluated by R).  

This article describes the process of creating a new model function. Before proceeding, take a minute and read our [guidelines on creating modeling packages](https://tidymodels.github.io/model-implementation-principles/) to understand the general themes and conventions that we use.  

## An example model

As an example, we'll create a function for _mixture discriminant analysis_. There are [a few packages](http://search.r-project.org/cgi-bin/namazu.cgi?query=%22mixture+discriminant%22&max=100&result=normal&sort=score&idxname=functions) that implement this but we'll focus on `mda::mda`:


```r
str(mda::mda)
#> function (formula = formula(data), data = sys.frame(sys.parent()), subclasses = 3, 
#>     sub.df = NULL, tot.df = NULL, dimension = sum(subclasses) - 1, eps = .Machine$double.eps, 
#>     iter = 5, weights = mda.start(x, g, subclasses, trace, ...), method = polyreg, 
#>     keep.fitted = (n * dimension < 5000), trace = FALSE, ...)
```

The main hyperparameter is the number of subclasses. We'll name our function `discrim_mixture`. 

## Aspects of models

Before proceeding, it helps to to review how parsnip categorizes models:

* The model _type_ is related to the structural aspect of the model. For example, the model type `linear_reg` represents linear models (slopes and intercepts) that model a numeric outcome. Other model types in the package are `nearest_neighbor`, `decision_tree`, and so on. 

* Within a model type is the _mode_, related to the modeling goal. Currently the two modes in the package are regression and classification. Some models have methods for both models (e.g. nearest neighbors) while others have only a single mode (e.g. logistic regression). 

* The computation _engine_ is a combination of the estimation method and the implementation. For example, for linear regression, one engine is `"lm"` which uses ordinary least squares analysis via the `lm()` function. Another engine is `"stan"` which uses the Stan infrastructure to estimate parameters using Bayes rule. 

When adding a model into parsnip, the user has to specify which modes and engines are used. The package also enables users to add a new mode or engine to an existing model. 

## The general process

The parsnip package stores information about the models in an internal environment object. The environment can be accessed via the function `get_model_env()`. The package includes a variety of functions that can get or set the different aspects of the models. 

If you are adding a new model from your own package, you can use these functions to add new entries into the model environment. 

### Step 1. Register the model, modes, and arguments

We will add the MDA model using the model type `discrim_mixture`. Since this is a classification method, we only have to register a single mode:


```r
library(tidymodels)
set_new_model("discrim_mixture")
set_model_mode(model = "discrim_mixture", mode = "classification")
set_model_engine(
  "discrim_mixture", 
  mode = "classification", 
  eng = "mda"
)
set_dependency("discrim_mixture", eng = "mda", pkg = "mda")
```

These functions should silently finish. There is also a function that can be used to show what aspects of the model have been added to parsnip: 


```r
show_model_info("discrim_mixture")
#> Information for `discrim_mixture`
#>  modes: unknown, classification 
#> 
#>  engines: 
#>    classification: mda
#> 
#>  no registered arguments.
#> 
#>  no registered fit modules.
#> 
#>  no registered prediction modules.
```

The next step would be to declare the main arguments to the model. These are declared independent of the mode.  To specify the argument, there are a few slots to fill in:

 * The name that parsnip uses for the argument. In general, we try to use non-jargony names for arguments (e.g. "penalty" instead of "lambda" for regularized regression). We recommend consulting [the model argument table available here](/find/parsnip/) to see if an existing argument name can be used before creating a new one. 
 
 * The argument name that is used by the underlying modeling function. 
 
 * A function reference for a _constructor_ that will be used to generate tuning parameter values. This should be a character vector with a named element called `fun` that is the constructor function. There is an optional element `pkg` that can be used to call the function using its namespace. If referencing functions from the dials package, quantitative parameters can have additional arguments in the list for `trans` and `range` while qualitative parameters can pass `values` via this list.  
 
 * A logical value for whether the argument can be used to generate multiple predictions for a single R object. For example, for boosted trees, if a model is fit with 10 boosting iterations, many modeling packages allow the model object to make predictions for any iterations less than the one used to fit the model. In general this is not the case so one would use `has_submodels = FALSE`. 
 
For `mda::mda()`, the main tuning parameter is `subclasses` which we will rewrite as `sub_classes`. 


```r
set_model_arg(
  model = "discrim_mixture",
  eng = "mda",
  parsnip = "sub_classes",
  original = "subclasses",
  func = list(pkg = "foo", fun = "bar"),
  has_submodel = FALSE
)
show_model_info("discrim_mixture")
#> Information for `discrim_mixture`
#>  modes: unknown, classification 
#> 
#>  engines: 
#>    classification: mda
#> 
#>  arguments: 
#>    mda: 
#>       sub_classes --> subclasses
#> 
#>  no registered fit modules.
#> 
#>  no registered prediction modules.
```

### Step 2. Create the model function

This is a fairly simple function that can follow a basic template. The main arguments to our function will be:

 * The mode. If the model can do more than one mode, you might default this to "unknown". In our case, since it is only a classification model, it makes sense to default it to that mode so that the users won't have to specify it. 
 
 * The argument names (`sub_classes` here). These should be defaulted to `NULL`.

A basic version of the function is:


```r
discrim_mixture <-
  function(mode = "classification",  sub_classes = NULL) {
    # Check for correct mode
    if (mode  != "classification") {
      rlang::abort("`mode` should be 'classification'")
    }
    
    # Capture the arguments in quosures
    args <- list(sub_classes = rlang::enquo(sub_classes))
    
    # Save some empty slots for future parts of the specification
    new_model_spec(
      "discrim_mixture",
      args = args,
      eng_args = NULL,
      mode = mode,
      method = NULL,
      engine = NULL
    )
  }
```

This is pretty simple since the data are not exposed to this function. 

{{% warning %}} We strongly suggest favoring `rlang::abort()` and `rlang::warn()` over `stop()` and `warning()`. The former return better traceback results and have safer defaults for handling call objects. {{%/ warning %}}

### Step 3. Add a fit module

Now that parsnip knows about the model, mode, and engine, we can give it the information on fitting the model for our engine. The information needed to fit the model is contained in another list. The elements are:

 * `interface` is a single character value that could be "formula", "data.frame", or "matrix". This defines the type of interface used by the underlying fit function (`mda::mda`, in this case). This helps the translation of the data to be in an appropriate format for the that function. 
 
 * `protect` is an optional list of function arguments that **should not be changeable** by the user. In this case, we probably don't want users to pass data values to these arguments (until the `fit()` function is called).
 
 * `func` is the package and name of the function that will be called. If you are using a locally defined function, only `fun` is required. 
 
 * `defaults` is an optional list of arguments to the fit function that the user can change, but whose defaults can be set here. This isn't needed in this case, but is described later in this document.

For the first engine:


```r
set_fit(
  model = "discrim_mixture",
  eng = "mda",
  mode = "classification",
  value = list(
    interface = "formula",
    protect = c("formula", "data"),
    func = c(pkg = "mda", fun = "mda"),
    defaults = list()
  )
)

show_model_info("discrim_mixture")
#> Information for `discrim_mixture`
#>  modes: unknown, classification 
#> 
#>  engines: 
#>    classification: mda
#> 
#>  arguments: 
#>    mda: 
#>       sub_classes --> subclasses
#> 
#>  fit modules:
#>  engine           mode
#>     mda classification
#> 
#>  no registered prediction modules.
```

We also set up the information on how the predictors should be handled. These options ensure that the data that parsnip gives to the underlying model allows for a model fit that is as similar as possible to what it would have produced directly.

 * `predictor_indicators` describes whether and how to create indicator/dummy variables from factor predictors. There are three options: `"none"` (do not expand factor predictors), `"traditional"` (apply the standard `model.matrix()` encodings), and `"one_hot"` (create the complete set including the baseline level for all factors). 
 
 * `compute_intercept` controls whether `model.matrix()` should include the intercept in its formula. This affects more than the inclusion of an intercept column. With an intercept, `model.matrix()` computes dummy variables for all but one factor level. Without an intercept, `model.matrix()` computes a full set of indicators for the first factor variable, but an incomplete set for the remainder.
 
 * `remove_intercept` removes the intercept column *after* `model.matrix()` is finished. This can be useful if the model function (e.g. `lm()`) automatically generates an intercept.

* `allow_sparse_x` specifies whether the model can accommodate a sparse representation for predictors during fitting and tuning.


```r
set_encoding(
  model = "discrim_mixture",
  eng = "mda",
  mode = "classification",
  options = list(
    predictor_indicators = "traditional",
    compute_intercept = TRUE,
    remove_intercept = TRUE,
    allow_sparse_x = FALSE
  )
)
```


### Step 4. Add modules for prediction

Similar to the fitting module, we specify the code for making different types of predictions. To make hard class predictions, the `class` object contains the details. The elements of the list are:

 * `pre` and `post` are optional functions that can preprocess the data being fed to the prediction code and to postprocess the raw output of the predictions. These won't be needed for this example, but a section below has examples of how these can be used when the model code is not easy to use. If the data being predicted has a simple type requirement, you can avoid using a `pre` function with the `args` below. 
 * `func` is the prediction function (in the same format as above). In many cases, packages have a predict method for their model's class but this is typically not exported. In this case (and the example below), it is simple enough to make a generic call to `predict()` with no associated package. 
 * `args` is a list of arguments to pass to the prediction function. These will most likely be wrapped in `rlang::expr()` so that they are not evaluated when defining the method. For mda, the code would be `predict(object, newdata, type = "class")`. What is actually given to the function is the parsnip model fit object, which includes a sub-object called `fit()` that houses the mda model object. If the data need to be a matrix or data frame, you could also use `newdata = quote(as.data.frame(newdata))` or similar. 

The parsnip prediction code will expect the result to be an unnamed character string or factor. This will be coerced to a factor with the same levels as the original data.  

To add this method to the model environment, a similar `set()` function is used:


```r
class_info <- 
  list(
    pre = NULL,
    post = NULL,
    func = c(fun = "predict"),
    args =
      # These lists should be of the form:
      # {predict.mda argument name} = {values provided from parsnip objects}
      list(
        # We don't want the first two arguments evaluated right now
        # since they don't exist yet. `type` is a simple object that
        # doesn't need to have its evaluation deferred. 
        object = quote(object$fit),
        newdata = quote(new_data),
        type = "class"
      )
  )

set_pred(
  model = "discrim_mixture",
  eng = "mda",
  mode = "classification",
  type = "class",
  value = class_info
)
```

A similar call can be used to define the class probability module (if they can be computed). The format is identical to the `class` module but the output is expected to be a tibble with columns for each factor level. 

As an example of the `post` function, the data frame created by `mda:::predict.mda()` will be converted to a tibble. The arguments are `x` (the raw results coming from the predict method) and `object` (the parsnip model fit object). The latter has a sub-object called `lvl` which is a character string of the outcome's factor levels (if any). 

We register the probability module. There is a template function that makes this slightly easier to format the objects:


```r
prob_info <-
  pred_value_template(
    post = function(x, object) {
      tibble::as_tibble(x)
    },
    func = c(fun = "predict"),
    # Now everything else is put into the `args` slot
    object = quote(object$fit),
    newdata = quote(new_data),
    type = "posterior"
  )

set_pred(
  model = "discrim_mixture",
  eng = "mda",
  mode = "classification",
  type = "prob",
  value = prob_info
)

show_model_info("discrim_mixture")
#> Information for `discrim_mixture`
#>  modes: unknown, classification 
#> 
#>  engines: 
#>    classification: mda
#> 
#>  arguments: 
#>    mda: 
#>       sub_classes --> subclasses
#> 
#>  fit modules:
#>  engine           mode
#>     mda classification
#> 
#>  prediction modules:
#>              mode engine     methods
#>    classification    mda class, prob
```

If this model could be used for regression situations, we could also add a "numeric" module. For `pred`, the model requires an unnamed numeric vector output (usually). 

Examples are [here](https://github.com/tidymodels/parsnip/blob/master/R/linear_reg_data.R) and [here](https://github.com/tidymodels/parsnip/blob/master/R/rand_forest_data.R). 


### Does it work? 

As a developer, one thing that may come in handy is the `translate()` function. This will tell you what the model's eventual syntax will be. 

For example:


```r
discrim_mixture(sub_classes = 2) %>%
  translate(engine = "mda")
#> Model Specification (classification)
#> 
#> Main Arguments:
#>   sub_classes = 2
#> 
#> Computational engine: mda 
#> 
#> Model fit template:
#> mda::mda(formula = missing_arg(), data = missing_arg(), subclasses = 2)
```

Let's try it on a data set from the modeldata package:


```r
data("two_class_dat", package = "modeldata")
set.seed(4622)
example_split <- initial_split(two_class_dat, prop = 0.99)
example_train <- training(example_split)
example_test  <-  testing(example_split)

mda_spec <- discrim_mixture(sub_classes = 2) %>% 
  set_engine("mda")

mda_fit <- mda_spec %>%
  fit(Class ~ ., data = example_train, engine = "mda")
mda_fit
#> parsnip model object
#> 
#> Fit time:  13ms 
#> Call:
#> mda::mda(formula = Class ~ ., data = data, subclasses = ~2)
#> 
#> Dimension: 2 
#> 
#> Percent Between-Group Variance Explained:
#>    v1    v2 
#>  82.6 100.0 
#> 
#> Degrees of Freedom (per dimension): 3 
#> 
#> Training Misclassification Error: 0.172 ( N = 783 )
#> 
#> Deviance: 671

predict(mda_fit, new_data = example_test, type = "prob") %>%
  bind_cols(example_test %>% select(Class))
#> # A tibble: 8 x 3
#>   .pred_Class1 .pred_Class2 Class 
#>          <dbl>        <dbl> <fct> 
#> 1       0.679         0.321 Class1
#> 2       0.690         0.310 Class1
#> 3       0.384         0.616 Class2
#> 4       0.300         0.700 Class1
#> 5       0.0262        0.974 Class2
#> 6       0.405         0.595 Class2
#> 7       0.793         0.207 Class1
#> 8       0.0949        0.905 Class2

predict(mda_fit, new_data = example_test) %>% 
 bind_cols(example_test %>% select(Class))
#> # A tibble: 8 x 2
#>   .pred_class Class 
#>   <fct>       <fct> 
#> 1 Class1      Class1
#> 2 Class1      Class1
#> 3 Class2      Class2
#> 4 Class2      Class1
#> 5 Class2      Class2
#> 6 Class2      Class2
#> 7 Class1      Class1
#> 8 Class2      Class2
```


## Add an engine

The process for adding an engine to an existing model is _almost_ the same as building a new model but simpler with fewer steps. You only need to add the engine-specific aspects of the model. For example, if we wanted to fit a linear regression model using M-estimation, we could only add a new engine. The code for the `rlm()` function in MASS is pretty similar to `lm()`, so we can copy that code and change the package/function names:


```r
set_model_engine("linear_reg", "regression", eng = "rlm")
set_dependency("linear_reg", eng = "rlm", pkg = "MASS")

set_fit(
  model = "linear_reg",
  eng = "rlm",
  mode = "regression",
  value = list(
    interface = "formula",
    protect = c("formula", "data", "weights"),
    func = c(pkg = "MASS", fun = "rlm"),
    defaults = list()
  )
)

set_encoding(
  model = "linear_reg",
  eng = "rlm",
  mode = "regression",
  options = list(
    predictor_indicators = "traditional",
    compute_intercept = TRUE,
    remove_intercept = TRUE,
    allow_sparse_x = FALSE
  )
)

set_pred(
  model = "linear_reg",
  eng = "rlm",
  mode = "regression",
  type = "numeric",
  value = list(
    pre = NULL,
    post = NULL,
    func = c(fun = "predict"),
    args =
      list(
        object = expr(object$fit),
        newdata = expr(new_data),
        type = "response"
      )
  )
)

# testing:
linear_reg() %>% 
  set_engine("rlm") %>% 
  fit(mpg ~ ., data = mtcars)
#> parsnip model object
#> 
#> Fit time:  3ms 
#> Call:
#> rlm(formula = mpg ~ ., data = data)
#> Converged in 8 iterations
#> 
#> Coefficients:
#> (Intercept)         cyl        disp          hp        drat          wt 
#>     17.8225     -0.2788      0.0159     -0.0254      0.4639     -4.1436 
#>        qsec          vs          am        gear        carb 
#>      0.6531      0.2498      1.4341      0.8594     -0.0108 
#> 
#> Degrees of freedom: 32 total; 21 residual
#> Scale estimate: 2.15
```

## Add parsnip models to another package

The process here is almost the same. All of the previous functions are still required but their execution is a little different. 

For parsnip to register them, that package must already be loaded. For this reason, it makes sense to have parsnip in the "Depends" category. 

The first difference is that the functions that define the model must be inside of a wrapper function that is called when your package is loaded. For our example here, this might look like: 


```r
make_discrim_mixture_mda <- function() {
  parsnip::set_new_model("discrim_mixture")

  parsnip::set_model_mode("discrim_mixture", "classification")

  # and so one...
}
```

This function is then executed when your package is loaded: 


```r
.onLoad <- function(libname, pkgname) {
  # This defines discrim_mixture in the model database
  make_discrim_mixture_mda()
}
```

For an example package that uses parsnip definitions, take a look at the [discrim](https://github.com/tidymodels/discrim) package.

{{% warning %}} To use a new model and/or engine in the broader tidymodels infrastructure, we recommend your model definition declarations (e.g. `set_new_model()` and similar) reside in a package. If these definitions are in a script only, the new model may not work with the tune package, for example for parallel processing. {{%/ warning %}}

It is also important for parallel processing support to **list the home package as a dependency**. If the `discrim_mixture()` function lived in a package called `mixedup`, include the line:

```r
set_dependency("discrim_mixture", eng = "mda", pkg = "mixedup")
```

Parallel processing requires this explicit dependency setting. When parallel worker processes are created, there is heterogeneity across technologies regarding which packages are loaded. Multicore methods on macOS and Linux will load all of the packages that were loaded in the main R process. However, parallel processing using psock clusters have no additional packages loaded. If the home package for a parsnip model is not loaded in the worker processes, the model will not have an entry in parsnip's internal database (and produce an error). 


## Your model, tuning parameters, and you

The tune package can be used to find reasonable values of model arguments via tuning. There are some S3 methods that are useful to define for your model. `discrim_mixture()` has one main tuning parameter: `sub_classes`. To work with tune it is _helpful_ (but not required) to use an S3 method called `tunable()` to define which arguments should be tuned and how values of those arguments should be generated. 

`tunable()` takes the model specification as its argument and returns a tibble with columns: 

* `name`: The name of the argument. 

* `call_info`: A list that describes how to call a function that returns a dials parameter object. 

* `source`: A character string that indicates where the tuning value comes from (i.e., a model, a recipe etc.). Here, it is just `"model_spec"`. 

* `component`: A character string with more information about the source. For models, this is just the name of the function (e.g. `"discrim_mixture"`). 

* `component_id`: A character string to indicate where a unique identifier is for the object. For a model, this is indicates the type of model argument (e.g. "main").  

The main piece of information that requires some detail is `call_info`. This is a list column in the tibble. Each element of the list is a list that describes the package and function that can be used to create a dials parameter object. 

For example, for a nearest-neighbors `neighbors` parameter, this value is just: 


```r
info <- list(pkg = "dials", fun = "neighbors")

# FYI: how it is used under-the-hood: 
new_param_call <- rlang::call2(.fn = info$fun, .ns = info$pkg)
rlang::eval_tidy(new_param_call)
#> # Nearest Neighbors (quantitative)
#> Range: [1, 10]
```

For `discrim_mixture()`, a dials object is needed that returns an integer that is the number of sub-classes that should be create. We can create a dials parameter function for this:


```r
sub_classes <- function(range = c(1L, 10L), trans = NULL) {
  new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(sub_classes = "# Sub-Classes"),
    finalize = NULL
  )
}
```

If this were in the dials package, we could use: 


```r
tunable.discrim_mixture <- function(x, ...) {
  tibble::tibble(
    name = c("sub_classes"),
    call_info = list(list(pkg = NULL, fun = "sub_classes")),
    source = "model_spec",
    component = "discrim_mixture",
    component_id = "main"
  )
}
```

Once this method is in place, the tuning functions can be used: 


```r
mda_spec <- 
  discrim_mixture(sub_classes = tune()) %>% 
  set_engine("mda")

set.seed(452)
cv <- vfold_cv(example_train)
mda_tune_res <- mda_spec %>%
  tune_grid(Class ~ ., cv, grid = 4)
show_best(mda_tune_res, metric = "roc_auc")
#> # A tibble: 4 x 7
#>   sub_classes .metric .estimator  mean     n std_err .config             
#>         <int> <chr>   <chr>      <dbl> <int>   <dbl> <chr>               
#> 1           2 roc_auc binary     0.890    10  0.0143 Preprocessor1_Model3
#> 2           3 roc_auc binary     0.889    10  0.0142 Preprocessor1_Model4
#> 3           6 roc_auc binary     0.884    10  0.0147 Preprocessor1_Model2
#> 4           8 roc_auc binary     0.881    10  0.0146 Preprocessor1_Model1
```



## Pro-tips, what-ifs, exceptions, FAQ, and minutiae

There are various things that came to mind while developing this resource.

**Do I have to return a simple vector for `predict` and `predict_class`?**

Previously, when discussing the `pred` information:

> For `pred`, the model requires an unnamed numeric vector output **(usually)**.

There are some models (e.g. `glmnet`, `plsr`, `Cubist`, etc.) that can make predictions for different models from the same fitted model object. We want to facilitate that here so, for these cases, the current convention is to return a tibble with the prediction in a column called `values` and have extra columns for any parameters that define the different sub-models. 

For example, if I fit a linear regression model via `glmnet` and get four values of the regularization parameter (`lambda`):


```r
linear_reg() %>%
  set_engine("glmnet", nlambda = 4) %>% 
  fit(mpg ~ ., data = mtcars) %>%
  multi_predict(new_data = mtcars[1:3, -1])
```

_However_, the API is still being developed. Currently, there is not an interface in the prediction functions to pass in the values of the parameters to make predictions with (`lambda`, in this case). 

**What do I do about how my model handles factors or categorical data?**

Some modeling functions in R create indicator/dummy variables from categorical data when you use a model formula (typically using `model.matrix()`), and some do not. Some examples of models that do _not_ create indicator variables include tree-based models, naive Bayes models, and multilevel or hierarchical models. The tidymodels ecosystem assumes a `model.matrix()`-like default encoding for categorical data used in a model formula, but you can change this encoding using `set_encoding()`. For example, you can set predictor encodings that say, "leave my data alone," and keep factors as is:


```r
set_encoding(
  model = "decision_tree",
  eng = "rpart",
  mode = "regression",
  options = list(
    predictor_indicators = "none",
    compute_intercept = FALSE,
    remove_intercept = FALSE
  )
)
```

{{% note %}} There are three options for `predictor_indicators`: 
- "none" (do not expand factor predictors)
- "traditional" (apply the standard `model.matrix()` encoding)
- "one_hot" (create the complete set including the baseline level for all factors)  {{%/ note %}}

To learn more about encoding categorical predictors, check out [this blog post](https://www.tidyverse.org/blog/2020/07/parsnip-0-1-2/#predictor-encoding-consistency).

**What is the `defaults` slot and why do I need it?**

You might want to set defaults that can be overridden by the user. For example, for logistic regression with `glm`, it make sense to default `family = binomial`. However, if someone wants to use a different link function, they should be able to do that. For that model/engine definition, it has:


```r
defaults = list(family = expr(binomial))
```

So that is the default:


```r
logistic_reg() %>% translate(engine = "glm")

# but you can change it:

logistic_reg() %>%
  set_engine("glm", family = expr(binomial(link = "probit"))) %>% 
  translate()
```

That's what `defaults` are for. 

Note that we wrapped `binomial` inside of `expr()`. If we didn't, it would substitute the results of executing `binomial()` inside of the expression (and that's a mess). 

**What if I want more complex defaults?**

The `translate` function can be used to check values or set defaults once the model's mode is known. To do this, you can create a model-specific S3 method that first calls the general method (`translate.model_spec()`) and then makes modifications or conducts error traps. 

For example, the ranger and randomForest package functions have arguments for calculating importance. One is a logical and the other is a string. Since this is likely to lead to a bunch of frustration and GitHub issues, we can put in a check:


```r
# Simplified version
translate.rand_forest <- function (x, engine, ...){
  # Run the general method to get the real arguments in place
  x <- translate.default(x, engine, ...)
  
  # Check and see if they make sense for the engine and/or mode:
  if (x$engine == "ranger") {
    if (any(names(x$method$fit$args) == "importance")) 
      if (is.logical(x$method$fit$args$importance)) 
        rlang::abort("`importance` should be a character value. See ?ranger::ranger.")
  }
  x
}
```

As another example, `nnet::nnet()` has an option for the final layer to be linear (called `linout`). If `mode = "regression"`, that should probably be set to `TRUE`. You couldn't do this with the `args` (described above) since you need the function translated first. 


**My model fit requires more than one function call. So....?**

The best course of action is to write wrapper so that it can be one call. This was the case with xgboost and keras. 

**Why would I preprocess my data?**

There might be non-trivial transformations that the model prediction code requires (such as converting to a sparse matrix representation, etc.)

This would **not** include making dummy variables and `model.matrix` stuff. The parsnip infrastructure already does that for you. 


**Why would I post-process my predictions?**

What comes back from some R functions may be somewhat... arcane or problematic. As an example, for xgboost, if you fit a multi-class boosted tree, you might expect the class probabilities to come back as a matrix (*narrator: they don't*). If you have four classes and make predictions on three samples, you get a vector of 12 probability values. You need to convert these to a rectangular data set. 

Another example is the predict method for ranger, which encapsulates the actual predictions in a more complex object structure. 

These are the types of problems that the post-processor will solve.  

**Are there other modes?**

Not yet but there will be. For example, it might make sense to have a different mode when doing risk-based modeling via Cox regression models. That would enable different classes of objects and those might be needed since the types of models don't make direct predictions of the outcome. 

If you have a suggestion, please add a [GitHub issue](https://github.com/tidymodels/parsnip/issues) to discuss it. 

 
## Session information


```
#> ─ Session info ───────────────────────────────────────────────────────────────
#>  setting  value                       
#>  version  R version 4.1.0 (2021-05-18)
#>  os       macOS Big Sur 11.4          
#>  system   aarch64, darwin20           
#>  ui       X11                         
#>  language (EN)                        
#>  collate  en_US.UTF-8                 
#>  ctype    en_US.UTF-8                 
#>  tz       America/Denver              
#>  date     2021-06-29                  
#> 
#> ─ Packages ───────────────────────────────────────────────────────────────────
#>  package    * version date       lib source        
#>  broom      * 0.7.8   2021-06-24 [1] CRAN (R 4.1.0)
#>  dials      * 0.0.9   2020-09-16 [1] CRAN (R 4.1.0)
#>  dplyr      * 1.0.7   2021-06-18 [1] CRAN (R 4.1.0)
#>  ggplot2    * 3.3.5   2021-06-25 [1] CRAN (R 4.1.0)
#>  infer      * 0.5.4   2021-01-13 [1] CRAN (R 4.1.0)
#>  mda        * 0.5-2   2020-06-29 [1] CRAN (R 4.1.0)
#>  modeldata  * 0.1.0   2020-10-22 [1] CRAN (R 4.1.0)
#>  parsnip    * 0.1.6   2021-05-27 [1] CRAN (R 4.1.0)
#>  purrr      * 0.3.4   2020-04-17 [1] CRAN (R 4.1.0)
#>  recipes    * 0.1.16  2021-04-16 [1] CRAN (R 4.1.0)
#>  rlang      * 0.4.11  2021-04-30 [1] CRAN (R 4.1.0)
#>  rsample    * 0.1.0   2021-05-08 [1] CRAN (R 4.1.0)
#>  tibble     * 3.1.2   2021-05-16 [1] CRAN (R 4.1.0)
#>  tidymodels * 0.1.3   2021-04-19 [1] CRAN (R 4.1.0)
#>  tune       * 0.1.5   2021-04-23 [1] CRAN (R 4.1.0)
#>  workflows  * 0.2.2   2021-03-10 [1] CRAN (R 4.1.0)
#>  yardstick  * 0.0.8   2021-03-28 [1] CRAN (R 4.1.0)
#> 
#> [1] /Library/Frameworks/R.framework/Versions/4.1-arm64/Resources/library
```


 
