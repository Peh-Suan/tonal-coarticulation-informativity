# R >= 4.4 (the notebook was run with R 4.4.1 under IRkernel)
pkgs <- c("lme4", "lmerTest", "dplyr", "readr", "tidyr", "ggplot2", "patchwork", "IRkernel",
          "brms", "posterior", "logspline", "knitr")   # last row: only for analysis/cross_language_models.ipynb
install.packages(setdiff(pkgs, rownames(installed.packages())))
# brms backend (cross_language_models.ipynb only): cmdstanr (faster) if available, otherwise rstan
if (!requireNamespace("cmdstanr", quietly = TRUE)) {
  install.packages("cmdstanr", repos = c("https://stan-dev.r-universe.dev", getOption("repos")))
  cmdstanr::install_cmdstan()
}
# IRkernel::installspec()   # once, to use R in Jupyter
