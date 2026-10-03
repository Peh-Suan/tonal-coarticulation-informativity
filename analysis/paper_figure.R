# Figure 1 of the paper: fixed-effect predictions with 95% Wald CIs.
# paper_figure_data(m, both) -> data.frame; paper_figure_plot(d, file) -> png/pdf.  Called from paper_model.ipynb.
# (a) target F0 by context F0; (b) coarticulation slope by word surprisal; (c) coarticulation slope by target-tone
# (solid) and context-tone (dashed) informativity, in bits over each language's central 95% range. Everything not
# shown is held at 0 (its mean); tone terms at 0 (sum coding).
paper_figure_data <- function(m, both) {
  b <- lme4::fixef(m); V <- as.matrix(vcov(m)); nm <- names(b)
  Lp <- function(at, l, p) { L <- setNames(numeric(length(b)), nm); L["(Intercept)"] <- 1
    vars <- c(names(at), "lang", "pos"); vals <- c(unlist(at), lang = l, pos = p)
    for (i in seq_along(nm)) { if (nm[i] == "(Intercept)") next
      comps <- strsplit(nm[i], ":", fixed = TRUE)[[1]]; if (all(comps %in% vars)) L[i] <- prod(vals[comps]) }
    L }
  ev <- function(L) c(sum(L * b), sqrt(drop(t(L) %*% V %*% L)))
  slope <- function(at, l, p) ev(Lp(modifyList(at, list(context_f0 = 1)), l, p) - Lp(modifyList(at, list(context_f0 = 0)), l, p))
  ST <- lapply(split(both, both$lang_lab), function(d) c(m = mean(d$informativity_raw), s = sd(d$informativity_raw),
          lo = unname(quantile(d$informativity_raw, 0.025)), hi = unname(quantile(d$informativity_raw, 0.975))))
  LANG <- c(TM = -0.5, TSM = 0.5); POS <- c(anticipatory = -0.5, carryover = 0.5)
  fo <- list(); add <- function(...) fo[[length(fo) + 1]] <<- data.frame(...)
  for (lg in names(LANG)) for (ps in names(POS)) { l <- LANG[[lg]]; p <- POS[[ps]]
    for (x in seq(-2, 2, 0.1)) { r <- ev(Lp(list(context_f0 = x), l, p)); add(panel = "a", language = lg, position = ps, which = "", x = x, est = r[1], se = r[2]) }
    for (x in seq(-2, 2, 0.1)) { r <- slope(list(unigram = x), l, p);   add(panel = "b", language = lg, position = ps, which = "", x = x, est = r[1], se = r[2]) }
    st <- ST[[lg]]
    for (x in seq(st[["lo"]], st[["hi"]], length.out = 60)) { z <- (x - st[["m"]]) / st[["s"]]
      r <- slope(list(informativity = z), l, p);         add(panel = "c", language = lg, position = ps, which = "Target tone", x = x, est = r[1], se = r[2])
      r <- slope(list(context_informativity = z), l, p); add(panel = "c", language = lg, position = ps, which = "Context tone", x = x, est = r[1], se = r[2]) } }
  fo <- do.call(rbind, fo); fo$l95 <- fo$est - 1.96 * fo$se; fo$u95 <- fo$est + 1.96 * fo$se; fo
}
paper_figure_plot <- function(d, file) {
  suppressPackageStartupMessages({ library(ggplot2); library(patchwork); library(dplyr) })
  d$language <- factor(d$language, c("TM", "TSM"))
  d$position <- factor(d$position, c("anticipatory", "carryover"), c("Anticipatory", "Carryover"))
  d$which <- factor(d$which, c("Target tone", "Context tone"))
  COL <- c(TM = "#0072B2", TSM = "#D55E00")
  th <- theme_minimal(base_size = 12) + theme(legend.position = "bottom", panel.grid.minor = element_blank(),
    axis.text = element_text(size = 10, colour = "grey20"), strip.text = element_text(size = 12, face = "bold"),
    plot.tag = element_text(face = "bold", size = 12), plot.margin = margin(2, 4, 2, 2))
  ylab <- "Coarticulation slope\n(target F0 on context F0)"
  mk <- function(pn, xl, yl, strip = FALSE) {
    g <- ggplot(filter(d, panel == pn), aes(x, est, colour = language, fill = language)) +
      geom_ribbon(aes(ymin = l95, ymax = u95), alpha = 0.25, colour = NA) + geom_line(linewidth = 0.9) +
      scale_colour_manual(values = COL, name = NULL) + scale_fill_manual(values = COL, name = NULL) +
      facet_wrap(~position) + labs(x = xl, y = yl) + th
    if (!strip) g <- g + theme(strip.text = element_blank()); g }
  pc <- ggplot(filter(d, panel == "c"), aes(x, est, colour = language, fill = language, linetype = which,
               group = interaction(language, which))) +
    geom_ribbon(aes(ymin = l95, ymax = u95), alpha = 0.12, colour = NA, show.legend = FALSE) + geom_line(linewidth = 0.9) +
    scale_colour_manual(values = COL, name = NULL) + scale_fill_manual(values = COL, name = NULL) +
    scale_linetype_manual(values = c("Target tone" = "solid", "Context tone" = "22"), name = NULL) +
    facet_wrap(~position) + labs(x = "Informativity (bits)", y = ylab) + th + theme(strip.text = element_blank()) +
    guides(colour = "none", fill = "none", linetype = guide_legend(override.aes = list(colour = "grey20"), keywidth = unit(2, "lines")))
  p <- mk("a", "Context F0 (z)", "Predicted target F0 (z)", TRUE) / mk("b", "Word surprisal (SD)", ylab) / pc +
    plot_layout(guides = "collect") + plot_annotation(tag_levels = list(c("(a)", "(b)", "(c)"))) &
    theme(legend.position = "bottom", legend.box.spacing = unit(0, "pt"), legend.margin = margin(0, 0, 0, 0))
  for (ext in c("png", "pdf")) ggsave(paste0(file, ".", ext), p, width = 6.5, height = 7.5, dpi = 300, bg = "white")
  invisible(p)
}
