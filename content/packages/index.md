---
title: Tidymodels packages
---

## Installation and use

* Install many of the packages in the tidymodels ecosystem by running `install.packages("tidymodels")`.

* Run `library(tidymodels)` to load the core packages and make them available in your current R session.

<div class="package-section">

<div class="package-section-info">

## Core tidymodels

  <p>The core tidymodels packages work together to enable a wide variety of modeling approaches:</p>
</div>

<div class="packages">
  <div class="package">
    <img class="package-image" src="/images/tidymodels.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://tidymodels.tidymodels.org/"> tidymodels </a></h3>
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

Go to package about the tidymodels packages at <http://tidymodels.tidymodels.org>.

## Specialized packages

The tidymodels framework also includes many other packages for more specialized use cases. They are not loaded automatically with `library(tidymodels)`, so you'll need to load each one with its own call to `library()`. 

<!--list them here-->
