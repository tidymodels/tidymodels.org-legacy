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
      <a href="https://tidymodels.tidymodels.org/" aria-hidden="true">Go to package ...</a></p>
    </div>
  </div>
  <div class="package">
    <img class="package-image" src="/images/rsample.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://rsample.tidymodels.org/">rsample</a></h3>
      <p>rsample provides infrastructure for efficient data splitting and resampling. <a href="https://rsample.tidymodels.org/" aria-hidden="true">Go to package ...</a></p>
    </div>
  </div>
  <div class="package">
    <img class="package-image" src="/images/parsnip.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://parsnip.tidymodels.org/"> parsnip </a></h3>
      <p>parsnip is a tidy, unified interface to models that can be used to try a range of models without getting bogged down in the syntactical minutiae of the underlying packages. <a href="https://parsnip.tidymodels.org/" aria-hidden="true">Go to package ...</a></p>
    </div>
  </div>  
  <div class="package">
    <img class="package-image" src="/images/recipes.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://recipes.tidymodels.org/"> recipes </a></h3>
      <p>recipes is a tidy interface to data pre-processing tools for feature engineering. <a href="https://recipes.tidymodels.org/" aria-hidden="true">Go to package ...</a></p>
    </div>
  </div>
  <div class="package">
    <img class="package-image" src="/images/workflows.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://workflows.tidymodels.org/"> workflows </a></h3>
      <p>workflows bundle your pre-processing, modeling, and post-processing together. <a href="https://workflows.tidymodels.org/" aria-hidden="true">Go to package ...</a></p>
    </div>
  </div> 
  <div class="package">
    <img class="package-image" src="/images/tune.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://tune.tidymodels.org/"> tune </a></h3>
      <p>tune helps you optimize the hyperparameters of your model and pre-processing steps. <a href="https://tune.tidymodels.org/" aria-hidden="true">Go to package ...</a></p>
    </div>
  </div>  
  <div class="package">
    <img class="package-image" src="/images/yardstick.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://yardstick.tidymodels.org/"> yardstick </a></h3>
      <p>yardstick measures the effectiveness of models using performance metrics. <a href="https://yardstick.tidymodels.org/" aria-hidden="true">Go to package ...</a></p>
    </div>
  </div>
  <div class="package">
    <img class="package-image" src="/images/broom.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://broom.tidymodels.org/"> broom </a></h3>
      <p>broom converts the information in common statistical R objects into user-friendly, predictable formats. 
      <a href="https://broom.tidymodels.org/" aria-hidden="true">Go to package ...</a></p>
    </div>
  </div>
  <div class="package">
    <img class="package-image" src="/images/dials.png" alt=""></img>
    <div class="package-info">
      <h3><a href="https://dials.tidymodels.org/"> dials </a></h3>
      <p>dials creates and manages tuning parameters and parameter grids. 
      <a href="https://dials.tidymodels.org/" aria-hidden="true">Go to package ...</a></p>
    </div>
  </div>  

</div>
</div>

Learn more about the tidymodels metapackage itself at <https://tidymodels.tidymodels.org/>.

## Specialized packages

The tidymodels framework also includes many other packages designed for specialized data analysis and modeling tasks. They are not loaded automatically with `library(tidymodels)`, so you'll need to load each one with its own call to `library()`. These packages include: 

### [Perform statistical analysis](/learn/statistics/)

* [infer](https://infer.netlify.com/) is a high-level API for tidyverse-friendly statistical inference.

* The [corrr](https://corrr.tidymodels.org/) package has tidy interfaces for working with correlation matrices.

### [Create robust models](/learn/models/)

* The [spatialsample](http://spatialsample.tidymodels.org/) package provides resampling functions and classes like rsample, but specialized for spatial data.

* parsnip also has additional packages that contain more model definitions. [discrim](https://discrim.tidymodels.org/) contains definitions for discriminant analysis models, [poissonreg](https://poissonreg.tidymodels.org/) provides definitions for Poisson regression models, [plsmod](https://plsmod.tidymodels.org/) enables linear projection models, and [rules](https://rules.tidymodels.org/) does the same for rule-based classification and regression models. [baguette](https://baguette.tidymodels.org/) creates ensemble models via bagging, and [multilevelmod](https://multilevelmod.tidymodels.org/) provides support for multilevel models (otherwise known as mixed models or hierarchical models). 

* There are several add-on packages for creating recipes. [embed](https://embed.tidymodels.org/) contains steps to create embeddings or projections of predictors. [textrecipes](https://textrecipes.tidymodels.org/) has extra steps for text processing, and [themis](https://themis.tidymodels.org/) can help alleviate class imbalance using sampling methods. 

* [tidypredict](https://tidypredict.tidymodels.org/) and [modeldb](https://modeldb.tidymodels.org/) can convert prediction equations to different languages (e.g. SQL) and fit some models in-database. 

### [Tune, compare, and work with your models](/learn/work/)

* To try out different workflows (i.e. bundles of pre-processor and model), [workflowsets](https://workflowsets.tidymodels.org/) lets you create sets of workflow objects for tuning and resampling.

* To integrate predictions from many models, the [stacks](https://stacks.tidymodels.org/) package provides tools for stacked ensemble modeling.

* The [finetune](https://finetune.tidymodels.org/) package contains some extra functions for model tuning that extend what is currently in the tune package.

* The [usemodels](https://usemodels.tidymodels.org/) package creates templates and automatically generates code to fit and tune models.

* [probably](https://probably.tidymodels.org/) has tools for post-processing class probability estimates.

* The [tidyposterior](https://tidyposterior.tidymodels.org/) package enables users to make formal statistical comparisons between models using resampling and Bayesian methods. 

* Some R objects become inconveniently large when saved to disk. The [butcher](https://butcher.tidymodels.org/) package can reduce the size of those objects by removing the sub-components. 

* To know whether the data that you are predicting are _extrapolations_ from the training set, [applicable](https://applicable.tidymodels.org/) can produce metrics that measure extrapolation. 

* [shinymodels](https://shinymodels.tidymodels.org/) lets you explore tuning or resampling results via a Shiny app.

### [Develop custom modeling tools](/learn/develop/)

* [hardhat](https://hardhat.tidymodels.org/) is a _developer-focused_ package that helps beginners create high-quality R packages for modeling. 
