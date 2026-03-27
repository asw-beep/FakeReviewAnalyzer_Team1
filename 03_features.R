# =============================================================================
# STAGE 3: Feature Extraction — TF-IDF + Handcrafted Features
# Review Trust Analyzer | CSS-321 | Group 1
# =============================================================================
# OBJECTIVE: Produce the main numerical feature representation for the
# classifier. Combine TF-IDF with handcrafted features that carry
# strong fake/genuine signal.
# =============================================================================

library(tidyverse)
library(tidytext)
library(Matrix)

# --------------------------------------------------------------------------
# 3.1  LOAD STAGE 2 OUTPUT
# --------------------------------------------------------------------------
cat("Loading Stage 2 data...\n")
amazon <- readRDS("dataset/amazon_stage2.rds")
fake   <- readRDS("dataset/fake_stage2.rds")
cat("Amazon:", nrow(amazon), "rows\n")
cat("Fake:  ", nrow(fake),   "rows\n")

# --------------------------------------------------------------------------
# 3.2  HANDCRAFTED FEATURES
# --------------------------------------------------------------------------
cat("\nComputing handcrafted features...\n")

# Applied to BOTH datasets
compute_handcrafted <- function(df) {
  df %>% mutate(
    review_length  = str_count(reviewText, "\\S+"),
    exclaim_count  = str_count(reviewText, "!"),
    caps_ratio     = ifelse(
      nchar(reviewText) > 0,
      str_count(reviewText, "[A-Z]") / nchar(reviewText),
      0
    ),
    avg_word_len   = ifelse(
      str_count(clean_text, "\\S+") > 0,
      nchar(gsub(" ", "", clean_text)) / str_count(clean_text, "\\S+"),
      0
    )
  )
}

amazon <- compute_handcrafted(amazon)
fake   <- compute_handcrafted(fake)

cat("Handcrafted features added: review_length, exclaim_count, caps_ratio, avg_word_len\n")

# --------------------------------------------------------------------------
# 3.3  TF-IDF MATRIX (on fake dataset — used for classifier training)
# --------------------------------------------------------------------------
cat("\nBuilding TF-IDF matrix on fake reviews dataset...\n")

# Tokenize clean_text into tidy format
fake_tokens <- fake %>%
  mutate(doc_id = row_number()) %>%
  select(doc_id, clean_text) %>%
  unnest_tokens(word, clean_text)

# Compute TF-IDF
fake_tfidf <- fake_tokens %>%
  count(doc_id, word) %>%
  bind_tf_idf(word, doc_id, n)

# Keep top 5000 terms by total TF-IDF weight
top_terms <- fake_tfidf %>%
  group_by(word) %>%
  summarise(total_tfidf = sum(tf_idf)) %>%
  arrange(desc(total_tfidf)) %>%
  slice_head(n = 5000) %>%
  pull(word)

cat("Top 5000 TF-IDF terms selected\n")

# Save vocabulary for later use (sentiment stage, classifier)
saveRDS(top_terms, "dataset/tfidf_vocabulary.rds")

# Build sparse document-term matrix
fake_dtm <- fake_tfidf %>%
  filter(word %in% top_terms) %>%
  select(doc_id, word, tf_idf) %>%
  pivot_wider(names_from = word, values_from = tf_idf, values_fill = 0)

cat("TF-IDF matrix shape:", nrow(fake_dtm), "docs x", ncol(fake_dtm) - 1, "terms\n")

# --------------------------------------------------------------------------
# 3.4  SUMMARY STATISTICS ON HANDCRAFTED FEATURES
# --------------------------------------------------------------------------
cat("\n=== Handcrafted Feature Summary (Fake Dataset) ===\n")
fake %>%
  group_by(label_name) %>%
  summarise(
    avg_length    = round(mean(review_length), 1),
    avg_exclaim   = round(mean(exclaim_count), 2),
    avg_caps      = round(mean(caps_ratio), 3),
    avg_word_len  = round(mean(avg_word_len), 2)
  ) %>%
  print()

# --------------------------------------------------------------------------
# 3.5  VISUALISATION — Feature Distributions by Label
# --------------------------------------------------------------------------
dir.create("figures", showWarnings = FALSE)

## Review length by label
p1 <- fake %>%
  filter(review_length < 200) %>%
  ggplot(aes(x = review_length, fill = label_name)) +
  geom_histogram(bins = 40, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("Genuine" = "#2ca25f", "Fake" = "#de2d26")) +
  labs(title = "Review Length Distribution by Label",
       x = "Word Count", y = "Count", fill = "Type") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"), legend.position = "top")

ggsave("figures/fig05_length_distribution.png", p1, width = 7, height = 5, dpi = 150)

## Exclamation count by label
p2 <- fake %>%
  filter(exclaim_count <= 10) %>%
  ggplot(aes(x = factor(exclaim_count), fill = label_name)) +
  geom_bar(position = "dodge", alpha = 0.85) +
  scale_fill_manual(values = c("Genuine" = "#2ca25f", "Fake" = "#de2d26")) +
  labs(title = "Exclamation Mark Count by Label",
       x = "Number of '!'", y = "Count", fill = "Type") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"), legend.position = "top")

ggsave("figures/fig06_exclaim_count.png", p2, width = 7, height = 5, dpi = 150)

cat("Saved: figures/fig05_length_distribution.png\n")
cat("Saved: figures/fig06_exclaim_count.png\n")

# --------------------------------------------------------------------------
# 3.6  SAVE FOR NEXT STAGE
# --------------------------------------------------------------------------
saveRDS(amazon,   "dataset/amazon_stage3.rds")
saveRDS(fake,     "dataset/fake_stage3.rds")
saveRDS(fake_dtm, "dataset/fake_tfidf_matrix.rds")

cat("\n=== Stage 3 Complete ===\n")
cat("Saved: dataset/amazon_stage3.rds\n")
cat("Saved: dataset/fake_stage3.rds\n")
cat("Saved: dataset/fake_tfidf_matrix.rds\n")
cat("Saved: dataset/tfidf_vocabulary.rds\n")
cat("Next: run 04_sentiment.R\n")