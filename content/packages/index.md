---
title: Tidymodels packages
---

# Installation and use

* Install many of the packages in the tidymodels ecosystem by running `install.packages("tidymodels")`.

* Run `library(tidymodels)` to load the core packages and make them available in your current R session.

<div class="package-section">

<div class="package-section-info">

# Core tidymodels

  <p>The core tidymodels packages work together to enable a wide variety of modeling approaches:</p>
</div>

<div class="packages">
  <div class="package">
    <img class="package-image" src="/images/tidymodels.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://tidymodels.github.io/tidymodels/"> tidymodels </a></h3>
      <p>tidymodels is a meta-package that installs and load the core packages listed below that you need for modeling and machine learning.
      <a href="https://tidymodels.github.io/tidymodels/" aria-hidden="true">Go to package ...</a></p>
    </div>
  </div>
  <div class="package">
    <img class="package-image" src="/images/rsample.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://tidymodels.github.io/rsample/">rsample</a></h3>
      <p>rsample provides infrastructure for efficient data splitting and resampling. <a href="https://tidymodels.github.io/rsample/" aria-hidden="true">Go to package ...</a></p>
    </div>
  </div>
  <div class="package">
    <img class="package-image" src="/images/parsnip.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://tidymodels.github.io/parsnip/"> parsnip </a></h3>
      <p>parsnip is a tidy, unified interface to models that can be used to try a range of models without getting bogged down in the syntactical minutiae of the underlying packages. <a href="https://tidymodels.github.io/parsnip/" aria-hidden="true">Go to package ...</a></p>
    </div>
  </div>  
  <div class="package">
    <img class="package-image" src="/images/recipes.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://tidymodels.github.io/rsample/"> recipes </a></h3>
      <p>recipes is a tidy interface to data pre-processing tools for feature engineering. <a href="https://tidymodels.github.io/rsample/" aria-hidden="true">Go to package ...</a></p>
    </div>
  </div>
  <div class="package">
    <img class="package-image" src="/images/placeholder.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://tidymodels.github.io/workflows/"> workflows </a></h3>
      <p>workflows bundle your pre-processing, modeling, and post-processing together. <a href="https://tidymodels.github.io/workflows/" aria-hidden="true">Go to package ...</a></p>
    </div>
  </div> 
  <div class="package">
    <img class="package-image" src="/images/tune.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://tidymodels.github.io/tune/"> tune </a></h3>
      <p>tune helps you optimize the hyperparameters of your model and pre-processing steps. <a href="https://tidymodels.github.io/tune/" aria-hidden="true">Go to package ...</a></p>
    </div>
  </div>  
  <div class="package">
    <img class="package-image" src="/images/yardstick.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://tidymodels.github.io/yardstick/"> yardstick </a></h3>
      <p>yardstick measures the effectiveness of models using performance metrics. <a href="https://tidymodels.github.io/yardstick/" aria-hidden="true">Go to package ...</a></p>
    </div>
  </div>
  <div class="package">
    <img class="package-image" src="/images/broom.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://broom.tidyverse.org/"> broom </a></h3>
      <p>broom converts the information in common statistical R objects into user-friendly, predictable formats. 
      <a href="https://broom.tidyverse.org/" aria-hidden="true">Go to package ...</a></p>
    </div>
  </div>
  <div class="package">
    <img class="package-image" src="/images/dials.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://dials.tidyverse.org/"> dials </a></h3>
      <p>dials creates and manages tuning parameters and parameter grids. 
      <a href="https://tidymodels.github.io/dials/" aria-hidden="true">Go to package ...</a></p>
    </div>
  </div>  

</div>
</div>

Learn more about the tidymodels metapackage itself at <https://tidymodels.github.io/tidymodels/>.

# Specialized packages

The tidymodels framework also includes many other packages designed for specialized data analysis and modeling tasks. They are not loaded automatically with `library(tidymodels)`, so you'll need to load each one with its own call to `library()`. These packages include: 

## Perform statistical analysis

* [infer](https://infer.netlify.com/) is a high-level API for tidyverse-friendly statistical inference.

* The [corrr](https://tidymodels.github.io/corrr/) package has tidy interfaces for working with correlation matrices.

## Create robust models

* parsnip also has additional packages that contain more model definitions. [discrim](https://tidymodels.github.io/discrim/) contains definitions for discriminant analysis models and [rules](https://tidymodels.github.io/rules/) does the same for rule-based classification and regression models. 

* There are several add-on packages for creating recipes. [embed](https://tidymodels.github.io/embed/) contains steps to create embeddings or projections of predictors. [textrecipes](https://tidymodels.github.io/textrecipes/) has extra steps for text processing, and [themis](https://tidymodels.github.io/themis/) can help alleviate class imbalance using sampling methods. 

* [tidypredict](https://tidymodels.github.io/tidypredict/) and [modeldb](https://tidymodels.github.io/modeldb/) can convert prediction equations to different languages (e.g. SQL) and fit some models in-database. 

## Tune, compare, and work with your models

* [probably](https://tidymodels.github.io/probably/) has tools for post-processing class probability estimates.

* The [tidyposterior](https://tidymodels.github.io/tidyposterior/) package enables users to make formal statistical comparisons between models using resampling and Bayesian methods. 

* Some R objects become inconveniently large when saved to disk. The [butcher](https://tidymodels.github.io/butcher/) package can reduce the size of those objects by removing the sub-components. 

* To know whether the data that you are predicting are _extrapolations_ from the training set, [applicable](https://tidymodels.github.io/applicable/) can produce metrics the measure extrapolation. 

## Develop custom modeling tools

* [hardhat](https://tidymodels.github.io/hardhat/) is a _developer-focused_ package that helps beginners create high-quality R packages for modeling. 
