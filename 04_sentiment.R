# =============================================================================
# STAGE 4: Sentiment Analysis using VADER (with live progress)
# Review Trust Analyzer | CSS-321 | Group 1
# =============================================================================

library(tidyverse)
library(vader)
library(scales)

cat("Loading Stage 3 data...\n")
amazon <- readRDS("dataset/amazon_stage3.rds")
fake   <- readRDS("dataset/fake_stage3.rds")
cat("Amazon:", nrow(amazon), "rows\n")
cat("Fake:  ", nrow(fake),   "rows\n")

# --------------------------------------------------------------------------
# HELPER: Score in batches with live progress
# --------------------------------------------------------------------------
score_in_batches <- function(texts, label = "reviews", batch_size = 1000) {
  n      <- length(texts)
  n_batches <- ceiling(n / batch_size)
  scores <- numeric(n)

  cat("  Scoring", n, label, "in", n_batches, "batches of", batch_size, "...\n")

  for (i in seq_len(n_batches)) {
    start_i <- (i - 1) * batch_size + 1
    end_i   <- min(i * batch_size, n)
    batch   <- texts[start_i:end_i]

    result        <- vader_df(batch)
    scores[start_i:end_i] <- result$compound

    pct <- round((end_i / n) * 100)
    cat(sprintf("  [%3d%%] Batch %d/%d done (%d reviews scored)\n",
                pct, i, n_batches, end_i))
    flush.console()
  }

  return(scores)
}

label_sentiment <- function(scores) {
  case_when(
    scores >= 0.05  ~ "positive",
    scores <= -0.05 ~ "negative",
    TRUE            ~ "neutral"
  )
}

# --------------------------------------------------------------------------
# Score Amazon (10K sample)
# --------------------------------------------------------------------------
set.seed(42)
amazon_sample <- amazon %>% sample_n(10000)

cat("\n--- Scoring Amazon sample (10K) ---\n")
amazon_sample <- amazon_sample %>%
  mutate(
    sentiment_score = score_in_batches(reviewText, "Amazon reviews"),
    sentiment_label = label_sentiment(sentiment_score)
  )

# Merge scores back into full amazon df
amazon <- amazon %>%
  mutate(sentiment_score = 0, sentiment_label = "neutral")
idx <- match(amazon_sample$Id, amazon$Id)
amazon$sentiment_score[idx] <- amazon_sample$sentiment_score
amazon$sentiment_label[idx] <- amazon_sample$sentiment_label

# --------------------------------------------------------------------------
# Score Fake dataset (full 40K)
# --------------------------------------------------------------------------
cat("\n--- Scoring Fake dataset (40K) ---\n")
fake <- fake %>%
  mutate(
    sentiment_score = score_in_batches(reviewText, "Fake reviews"),
    sentiment_label = label_sentiment(sentiment_score)
  )

# --------------------------------------------------------------------------
# Rating mismatch feature
# --------------------------------------------------------------------------
add_mismatch <- function(df) {
  df %>% mutate(
    rating_mismatch = case_when(
      overall >= 4 & sentiment_score <= -0.05 ~ 1L,
      overall <= 2 & sentiment_score >= 0.05  ~ 1L,
      TRUE ~ 0L
    )
  )
}
amazon <- add_mismatch(amazon)
fake   <- add_mismatch(fake)

# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
cat("\n=== Sentiment Distribution (Amazon 10K sample) ===\n")
print(table(amazon_sample$sentiment_label))

cat("\n=== Sentiment Distribution (Fake Dataset) ===\n")
print(table(fake$sentiment_label))

five_star <- amazon_sample %>% filter(overall == 5)
agreement <- mean(five_star$sentiment_label == "positive") * 100
cat("\n5-star reviews labelled positive by VADER:", round(agreement, 1), "%\n")
cat("(Target >= 85%)\n")

cat("\n=== Rating Mismatch by Fake/Genuine ===\n")
fake %>%
  group_by(label_name) %>%
  summarise(mismatch_rate = round(mean(rating_mismatch) * 100, 1)) %>%
  print()

# --------------------------------------------------------------------------
# Visualisations
# --------------------------------------------------------------------------
dir.create("figures", showWarnings = FALSE)

p1 <- amazon_sample %>%
  count(sentiment_label) %>%
  mutate(pct = n / sum(n),
         sentiment_label = factor(sentiment_label, levels = c("positive","neutral","negative"))) %>%
  ggplot(aes(x = sentiment_label, y = n, fill = sentiment_label)) +
  geom_col(width = 0.5, show.legend = FALSE) +
  geom_text(aes(label = paste0(comma(n), "\n(", percent(pct, 0.1), ")")), vjust = -0.3, size = 4) +
  scale_fill_manual(values = c("positive"="#2ca25f","neutral"="#f0a500","negative"="#de2d26")) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Amazon Reviews: Sentiment Distribution (10K sample)", x = "Sentiment", y = "Count") +
  theme_minimal(base_size = 13) + theme(plot.title = element_text(face = "bold"))
ggsave("figures/fig07_sentiment_distribution.png", p1, width = 7, height = 5, dpi = 150)

p2 <- amazon_sample %>%
  count(overall, sentiment_label) %>%
  group_by(overall) %>%
  mutate(pct = n / sum(n)) %>%
  ggplot(aes(x = sentiment_label, y = factor(overall), fill = pct)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = percent(pct, 1)), size = 3.5, color = "white", fontface = "bold") +
  scale_fill_gradient(low = "#deebf7", high = "#08519c", labels = percent) +
  labs(title = "Star Rating vs Sentiment Label", x = "VADER Sentiment", y = "Star Rating", fill = "% of row") +
  theme_minimal(base_size = 13) + theme(plot.title = element_text(face = "bold"))
ggsave("figures/fig08_rating_sentiment_heatmap.png", p2, width = 7, height = 5, dpi = 150)

p3 <- fake %>%
  ggplot(aes(x = sentiment_score, fill = label_name)) +
  geom_density(alpha = 0.6) +
  scale_fill_manual(values = c("Genuine"="#2ca25f","Fake"="#de2d26")) +
  labs(title = "Sentiment Score: Fake vs Genuine", x = "VADER Compound Score", y = "Density", fill = "Type") +
  theme_minimal(base_size = 13) + theme(plot.title = element_text(face = "bold"), legend.position = "top")
ggsave("figures/fig09_sentiment_by_label.png", p3, width = 7, height = 5, dpi = 150)

cat("Saved: figures/fig07, fig08, fig09\n")

saveRDS(amazon, "dataset/amazon_stage4.rds")
saveRDS(fake,   "dataset/fake_stage4.rds")

cat("\n=== Stage 4 Complete ===\n")
cat("Saved: dataset/amazon_stage4.rds\n")
cat("Saved: dataset/fake_stage4.rds\n")
cat("Next: run 05_topic_modeling.R\n")