library(dplyr)
library(purrr)
library(tidyr)
library(fs)
library(rlang)
library(pkgdown)
library(glue)
library(stringr)

get_pkgdown_urls <-
  function(x, pth = tempdir(), fltr = "[\\s\\S]*") {
    pkg_name <- x$pkg
    
    x <- mutate(x, base_url = paste0(base_url, "reference/"))
    
    base_url <- paste0(x$base_url, "{file}")
    null_res <-
      tibble::tibble(
        topic = rlang::na_chr,
        title = rlang::na_chr,
        url = rlang::na_chr,
        pkg = pkg_name
      )
    src_file <-
      download.packages(pkg_name, destdir = pth, repos = "https://cran.rstudio.com/")
    if (nrow(src_file) != length(pkg_name)) {
      found <- src_file[, 1]
      lost <- pkg_name[!(pkg_name %in% found)]
      lost <- paste0("'", lost, "'", collapse = ", ")
      msg <- paste("packages", lost, "were not downloaded")
      rlang::abort(msg)
    }
    untar_res <- purrr::map_int(src_file[, 2], untar, exdir = pth)
    if (any(untar_res != 0)) {
      msg <- paste("packages",
                   paste0("'", pkg_name[untar_res != 0], "'", collapse = ", "),
                   "did not unpack correctly")
      rlang::abort(msg)
    }
    
    topic_info <-
      purrr::map(pkg_name, ~ pkgdown::as_pkgdown(fs::path(pth, .x))) %>%
      purrr::map( ~ pluck(.x, "topics")) %>%
      purrr::map2(pkg_name, ~ .x %>% mutate(pkg = .y)) %>%
      bind_rows()  %>%
      unnest(cols = c(alias)) %>%
      full_join(x, by = "pkg") %>% 
      mutate(url = map2_chr(base_url, file_out, paste0),
             topic = alias) %>%
      dplyr::select(topic, alias, title, url, pkg) %>%
      mutate(title = str_replace(title, "\\n", " ")) %>%
      dplyr::filter(str_detect(topic, fltr)) %>%
      na.omit() %>%
      dplyr::filter(
        str_detect(topic, "reexport", negate = TRUE),
        str_detect(topic, "-package$", negate = TRUE),
        str_detect(title, "^Internal", negate = TRUE),
        str_detect(title, "^Tidy eval", negate = TRUE),
        topic != "_PACKAGE",
        title != "Pipe",
        topic != "%>%",
        title != "Objects exported from other packages"
      ) %>%
      dplyr::arrange(topic, pkg) %>%
      mutate(topic = paste0("<a href='", url, 
                            "'  target='_blank'><tt>", 
                            topic, "</tt></a>")
      ) %>%
      dplyr::select(topic, package = pkg, title, alias)
    
    topic_info
  }
