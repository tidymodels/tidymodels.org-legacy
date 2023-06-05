# ------------------------------------------------------------------------------
# Make data sets for function reference searches. Run this offline to refresh
# data objects.

library(tidymodels)
library(glue)
library(utils)
library(revdepcheck)
library(fs)
library(pkgdown)
library(urlchecker)

# ------------------------------------------------------------------------------

tidymodels_prefer()
theme_set(theme_bw())
options(pillar.advice = FALSE, pillar.min_title_chars = Inf)

# ------------------------------------------------------------------------------
# Use the pkgdown package to parse the source files and put them into a usable format

# TODO find a better way to figure out how to find the true "check_" recipe operations
# from just the source files

get_pkg_info <- function(pkg, pth = tempdir(), keep_internal = FALSE, pattern = NULL) {
  src_file <-
    download.packages(pkg,
                      destdir = pth,
                      repos = "https://cran.rstudio.com/",
                      quiet = TRUE)
  if (nrow(src_file) != length(pkg)) {
    return(NULL)
    rlang::warn(glue::glue("package {pkg} was not downloaded"))
  }
  pkg_path <- fs::path(pth, pkg)
  on.exit(fs::dir_delete(pkg_path))

  untar_res <- purrr::map_int(src_file[, 2], untar, exdir = pth)
  fs::file_delete(src_file[, 2])
  if (any(untar_res != 0)) {
    rlang::abort(glue::glue("package {pkg} did not unpack correctly"))
  }
  pkg_info <- pkgdown::as_pkgdown(pkg_path)
  res <- pkg_info$topics
  if (!keep_internal) {
    res <- dplyr::filter(res, !internal)
  }
  res <-
    res %>%
    dplyr::select(file_out, functions = alias, title) %>%
    tidyr::unnest(functions) %>%
    mutate(package = pkg, all_urls = list(pkg_info$desc$get_urls())) %>%
    relocate(package, all_urls)
  if (!is.null(pattern)) {
    res <- dplyr::filter(res, grepl(pattern, functions))
  }
  res
}

# See if any of the urls appear to correspond to the _standard_ pkgdown structure.
# Is so, link to the specific pkgdown html package, otherwise link to the first
# url or, if there are none listed, the canonical CRAN page link.
# We use an internal function in urlchecker to essentially ping the potential url

sort_out_urls <- function(x) {
  test_urls <-
    x %>%
    group_by(package) %>%
    slice(1) %>%
    ungroup() %>%
    unnest(all_urls) %>%
    mutate(
      URL = map_chr(all_urls, ~ glue("{.x[[1]]}/reference/index.html")),
      URL = gsub("//", "/", URL, fixed = TRUE)
    ) %>%
    select(URL, Parent = functions, package, all_urls)
  url_check_fails <-
    urlchecker:::tools$check_url_db(test_urls) %>%
    dplyr::select(URL)
  pkgdown_urls <-
    test_urls %>%
    anti_join(url_check_fails, by = "URL") %>%
    select(package, pkgdown_url = all_urls) %>%
    group_by(package) %>%
    slice(1) %>%
    ungroup()
  x %>%
    left_join(pkgdown_urls, by = "package") %>%
    mutate(
      first_url = map_chr(all_urls, ~ .x[1]),
      first_url = ifelse(is.na(first_url),
                         glue("https://cran.r-project.org/package={package}"),
                         first_url),
      base_url = ifelse(is.na(pkgdown_url),
                        first_url,
                        pkgdown_url),
      url = ifelse(!is.na(pkgdown_url),
                   glue("{pkgdown_url}/reference/{file_out}"),
                   base_url),
      topic = glue("<a href='{url}' target='_blank'><tt>{functions}</tt></a>")
    ) %>%
    dplyr::select(title, functions, topic, package) %>%
    mutate(package = as.factor(package)) %>%
    filter(!grepl("deprecated", tolower(title))) %>%
    arrange(tolower(gsub("[[:punct:]]", "", title)))
}

# ------------------------------------------------------------------------------

broom_pkgs <- revdepcheck::cran_revdeps("broom",    dependencies = c("Depends", "Imports"))
generics_pkgs <- revdepcheck::cran_revdeps("generics", dependencies = "Imports")

broom_pkgs <- sort(unique(c(broom_pkgs, generics_pkgs)))
excl <- c("hydrorecipes", "healthcareai")
broom_pkgs <- broom_pkgs[!(broom_pkgs %in% excl)]

broom_functions <-
  map_dfr(
    broom_pkgs,
    get_pkg_info,
    pattern = "(^tidy\\.)|(^glance\\.)|(^augment\\.)",
    .progress = TRUE
  ) %>%
  sort_out_urls() %>%
  select(-functions)

save(
  broom_functions,
  file = "find/broom/broom_functions.RData",
  compress = TRUE)

# ------------------------------------------------------------------------------

recipe_pkgs <- revdepcheck::cran_revdeps("recipes", dependencies = c("Depends", "Imports"))

recipe_pkgs <- sort(unique(c(recipe_pkgs)))
excl <- c("hydrorecipes", "healthcareai")
recipe_pkgs <- recipe_pkgs[!(recipe_pkgs %in% excl)]


recipe_functions <-
  map_dfr(
    recipe_pkgs,
    get_pkg_info,
    pattern = "^step_",
    .progress = TRUE
  )  %>%
  sort_out_urls() %>%
  select(-functions)

save(
  recipe_functions,
  file = "find/recipes/recipe_functions.RData",
  compress = TRUE)

# ------------------------------------------------------------------------------

all_tm <-
  c("agua", "applicable", "baguette", "brulee", "broom", "butcher",
    "censored", "corrr", "dials", "discrim", "embed", "finetune",
    "hardhat", "infer", "modeldata", "modeldb",
    "modelenv", "multilevelmod", "parsnip", "plsmod", "poissonreg",
    "probably", "recipes", "rsample", "rules", "shinymodels", "spatialsample",
    "stacks", "textrecipes", "themis", "tidyclust", "tidymodels",
    "tidyposterior", "tidypredict", "tune", "usemodels", "workflows",
    "workflowsets", "yardstick")


tidymodels_functions <-
  map_dfr(
    all_tm,
    get_pkg_info,
    .progress = TRUE
  ) %>%
  sort_out_urls() %>%
  filter(grepl("^\\.", functions)) %>%
  select(-functions)

save(
  tidymodels_functions,
  file = "find/all/tidymodels_functions.RData",
  compress = TRUE)

# ------------------------------------------------------------------------------

parsnip_pkgs <- revdepcheck::cran_revdeps("parsnip", dependencies = c("Depends", "Imports"))
parsnip_pkgs <- c(parsnip_pkgs, "parsnip")
# These ignore the tidymodels design principles and/or don't work with the broader ecosystem
# or we don't don't have any models in them
excl <- c("additive", "bayesian", "SSLR", "workflowsets", "workflows", "tune",
          "tidymodels", "shinymodels", "stacks")
parsnip_pkgs <- parsnip_pkgs[!(parsnip_pkgs %in% excl)]

# Load them then get the model data base
loaded <- map_lgl(parsnip_pkgs, ~ suppressPackageStartupMessages(require(.x, character.only = TRUE)))
table(loaded)

# h2o overwrites soooo many functions; this may take a few minutes
conflicted::conflict_prefer_all("base", loser = "h2o", quiet = TRUE)

model_list <-
  map_dfr(get_from_env("models"), ~ get_from_env(.x) %>% mutate(model = .x)) %>%
  mutate(
    mode = factor(mode, levels = c("classification", "regression", "censored regression"))
  ) %>%
  group_nest(model, engine) %>%
  mutate(
    modes = map_chr(data, ~ paste0(sort(.x$mode), collapse = ", ")),
    functions = glue("details_{model}_{engine}")
  ) %>%
  select(-data)

parsnip_model_info <-
  map_dfr(
    parsnip_pkgs,
    get_pkg_info,
    keep_internal = TRUE,
    .progress = TRUE
  ) %>%
  sort_out_urls()

# Split model/engine combinations by whether they have "details" pages. Link to
# the details pages whenever possible.

has_details <-
  parsnip_model_info %>%
  filter(grepl("^details_", functions)) %>%
  inner_join(model_list, by = "functions") %>%
  mutate(topic = gsub("<tt>details_", "<tt>", topic))

no_details <-
  model_list %>%
  anti_join(has_details %>% select(model, engine), by = c("model", "engine")) %>%
  mutate(functions = model) %>%
  inner_join(parsnip_model_info, by = "functions")

parsnip_models <-
  no_details %>%
  select(title, model, engine, topic, modes, package) %>%
  bind_rows(
    has_details %>%
      select(title, model, engine, topic, modes, package)
  ) %>%
  mutate(
    model = paste0("<code>",  model, "</code>"),
    engine = paste0("<code>",  engine, "</code>"),
    title = gsub("General Interface for ", "", title)
  ) %>%
  arrange(model, engine) %>%
  select(title, model, engine, topic, modes, package)

save(
  parsnip_models,
  file = "find/parsnip/parsnip_models.RData",
  compress = TRUE)


