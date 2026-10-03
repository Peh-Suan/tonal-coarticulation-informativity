# Contrasts on the coarticulation slope reported in the paper (beta, SE, 95% CI, Satterthwaite df, p).
# paper_contrasts(m) -> data.frame; called from paper_model.ipynb.
# Coding: lang = -0.5 TM / +0.5 TSM; pos = -0.5 anticipatory / +0.5 carryover; every other predictor at 0 (its mean).
paper_contrasts <- function(m) {
  stopifnot(inherits(m, "lmerModLmerTest"))
  b <- lme4::fixef(m); nm <- names(b)
  key  <- function(x) paste(sort(strsplit(x, ":", fixed = TRUE)[[1]]), collapse = ":")
  nkey <- vapply(nm, key, "")
  # L vector for d(target_f0)/d(context_f0), optionally modified by `eff`, at lang l and pos p
  Lvec <- function(eff = NULL, l = 0, p = 0) {
    L <- setNames(numeric(length(b)), nm); base <- c("context_f0", eff)
    add <- function(extra, w) { k <- key(paste(c(base, extra), collapse = ":")); i <- which(nkey == k)
      if (length(i) != 1) stop("term not found: ", k); L[i] <<- L[i] + w }
    add(NULL, 1); add("pos", p); add("lang", l); add(c("pos", "lang"), p * l); L }
  TM <- -0.5; TSM <- 0.5; A <- -0.5; C <- 0.5; lab <- function(l) ifelse(l < 0, "TM", "TSM"); pl <- function(p) ifelse(p < 0, "a", "c")
  rows <- list()
  one <- function(effect, contrast, L) {
    r <- lmerTest::contest(m, matrix(L, nrow = 1), joint = FALSE, confint = TRUE, level = 0.95)
    rows[[length(rows) + 1]] <<- data.frame(effect, contrast, estimate = r$Estimate, se = r$`Std. Error`,
      l95 = r$lower, u95 = r$upper, df = r$df, t = r$`t value`, p = r$`Pr(>|t|)`) }
  for (eff in list(NULL, "unigram", "gain", "informativity", "context_informativity")) {
    e <- if (is.null(eff)) "slope" else eff
    for (pp in c(A, C)) for (ll in c(TM, TSM)) one(e, sprintf("%s | %s", lab(ll), pl(pp)), Lvec(eff, ll, pp))
    for (ll in c(TM, TSM)) one(e, sprintf("%s | average", lab(ll)), Lvec(eff, ll, 0))
    for (pp in c(A, C)) one(e, sprintf("both | %s", pl(pp)), Lvec(eff, 0, pp))
    one(e, "both | average", Lvec(eff, 0, 0))
    for (pp in c(A, C)) one(e, sprintf("TSM - TM | %s", pl(pp)), Lvec(eff, TSM, pp) - Lvec(eff, TM, pp))
    one(e, "TSM - TM | average", Lvec(eff, TSM, 0) - Lvec(eff, TM, 0))
    for (ll in c(TM, TSM)) one(e, sprintf("%s | c - a", lab(ll)), Lvec(eff, ll, C) - Lvec(eff, ll, A))
    one(e, "both | c - a", Lvec(eff, 0, C) - Lvec(eff, 0, A))
    one(e, "(TSM - TM) | c - a", (Lvec(eff, TSM, C) - Lvec(eff, TM, C)) - (Lvec(eff, TSM, A) - Lvec(eff, TM, A)))
  }
  out <- do.call(rbind, rows); out[, 3:9] <- lapply(out[, 3:9], function(x) signif(x, 4)); out
}
