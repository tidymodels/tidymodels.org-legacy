Although the site is building `blogdown` , we've bastardized the framework and use it in a slightly irregular way. Normally, a blog site would only execute and render articles that are new. In out case, when a new CRAN version of a package codes out, we want to re-run all of the articles. Otherwise, we'd potentially have a lot of "dead" articles that have results but cannot be reproduced with the most up-to-date versions of the packages. 

This happens because the GitHub repo does not contain any article results (e.g. `index.markdown` or `index.html`) so. don't commit any. 

# Deploying

To re-build the site:

* Make a branch (or fork) and make changes/test locally. 
* It is advisable to do a fresh install of tidymodels packages from CRAN to make sure that your results will be consistent with what the website will deploy. 
* If a new CRAN version of a package exists, use `renv` to add it by either using

```R
renv::snapshot()

# or

renv::install("parsnip")
```

* Commit the changes to the `Rmd` file and/or snapshot lock file. **Do not** commit the results of the articles (except for special cases).

* Create a PR.

The site should rebuild when the PR is pushed to the master branch. 

