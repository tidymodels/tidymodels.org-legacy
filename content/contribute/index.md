---
title: How to contribute to tidymodels
---

The ecosystem of tidymodels packages would not be possible without the contributions of the R community. No matter your current skills, it's possible to contribute back to tidymodels. Contributions are guided by our design goals.

## Design goals

The goals of tidymodels packages are to:

 * Encourage empirical validation and good statistical practice.

 * Smooth out heterogeneous interfaces.
 
 * Establish highly reusable infrastructure.

 * Enable a wider variety of methodologies.

 * Help package developers quickly build high quality model packages of their own.

These goals are guided by our [principles for creating modeling packages](https://tidymodels.github.io/model-implementation-principles/). 

**What are different ways _you_ can contribute?**

## Answer questions

You can help others use and learn tidymodels by answering questions on the [RStudio community site](https://community.rstudio.com/tag/tidymodels), [Stack Overflow](https://stackoverflow.com/questions/tagged/tidymodels?sort=newest), and [Twitter](https://twitter.com/search?q=%23tidymodels&f=live). Many people asking for help with tidymodels don't know what a [reprex](https://www.tidyverse.org/help#reprex) is or how to craft one. Acknowledging an individual's problem, showing them how to build a reprex, and pointing them to helpful resources are all enormously beneficial, even if you don't immediately solve their problem.

Remember that while you might have seen a problem a hundred times before, it's new to the person asking it. Be patient, polite, and empathic.

## File issues

If you've found a bug, first create a minimal [reprex](https://www.tidyverse.org/help#reprex). Spend some time working to make it as minimal as possible; the more time you spend doing this, the easier it is to fix the bug. When your reprex is ready, file it on the [GitHub repo](https://github.com/tidymodels/) of the appropriate package. 

The tidymodels team often focuses on one package at a time to reduce context switching and be more efficient. We may not address each issue right away, but we will use the reprex you create to understand your problem when it is time to focus on that package.

## Contribute documentation

Documentation is a high priority for tidymodels, and pull requests to correct or improve documentation are welcome. The most important thing to know is that tidymodels packages use [roxygen2](https://roxygen2.r-lib.org/); this means that documentation is found in the R code close to the source of each function. There are some special tags, but most tidymodels packages now use markdown in the documentation. This makes it particularly easy to get started!


## Contribute code

If you are a more experienced R programmer, you may have the inclination, interest, and ability to contribute directly to package development. Before you submit a pull request on a tidymodels package, always file an issue and confirm the tidymodels team agrees with your idea and is happy with your basic proposal.

In tidymodels packages, we use the [tidyverse style guide](https://style.tidyverse.org/) which will make sure that your new code and documentation matches the existing style. This makes the review process much smoother.

The tidymodels packages are explicitly built to support the creation of other modeling packages, and we would love to hear about what you build yourself! Check out our learning resources for [developing custom modeling tools](/learn/develop/).

