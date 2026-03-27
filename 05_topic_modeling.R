# =============================================================================
# STAGE 5: Topic Modeling using LDA
# Review Trust Analyzer | CSS-321 | Group 1
# =============================================================================
# OBJECTIVE: Discover the latent themes customers discuss across all reviews.
# Assign each review a dominant topic.
# =============================================================================

library(tidyverse)
library(tidytext)
library(topicmodels)
library(scales)

cat("Loading Stage 4 data...\n")
amazon <- readRDS("dataset/amazon_stage4.rds")
cat("Amazon:", nrow(amazon), "rows\n")

# --------------------------------------------------------------------------
# 5.1  BUILD DOCUMENT-TERM MATRIX FOR LDA
# --------------------------------------------------------------------------
cat("\nBuilding Document-Term Matrix...\n")

# Use a 10K sample for LDA — fast and sufficient for topic discovery
set.seed(42)
amazon_lda <- amazon %>% sample_n(10000) %>% mutate(doc_id = row_number())

dtm <- amazon_lda %>%
  select(doc_id, clean_text) %>%
  unnest_tokens(word, clean_text) %>%
  count(doc_id, word) %>%
  filter(n > 1) %>%                    # remove words appearing once
  cast_dtm(doc_id, word, n)

cat("DTM shape:", nrow(dtm), "docs x", ncol(dtm), "terms\n")

# Remove empty rows (documents with no terms after filtering)
row_totals <- slam::row_sums(dtm)
dtm <- dtm[row_totals > 0, ]
cat("DTM after removing empty docs:", nrow(dtm), "docs\n")

# --------------------------------------------------------------------------
# 5.2  TRAIN LDA MODEL
# --------------------------------------------------------------------------
cat("\nTraining LDA model (8 topics, 10 passes)...\n")
cat("This takes 2-4 minutes...\n")

set.seed(42)
lda_model <- LDA(dtm, k = 8, method = "Gibbs",
                 control = list(seed = 42, iter = 500, burnin = 100))

cat("LDA training complete.\n")

# --------------------------------------------------------------------------
# 5.3  EXTRACT AND PRINT TOP TERMS PER TOPIC
# --------------------------------------------------------------------------
cat("\n=== Top 10 Keywords per Topic ===\n")

top_terms <- tidy(lda_model, matrix = "beta") %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  arrange(topic, desc(beta))

for (t in 1:8) {
  words <- top_terms %>% filter(topic == t) %>% pull(term)
  cat(sprintf("Topic %d: %s\n", t, paste(words, collapse = ", ")))
}

# Manual topic labels — update these after reading the keywords above
topic_labels <- c(
  "1" = "Product Quality",
  "2" = "Food & Taste",
  "3" = "Packaging & Delivery",
  "4" = "Value for Money",
  "5" = "Health & Ingredients",
  "6" = "Customer Experience",
  "7" = "Pet Food",
  "8" = "Coffee & Beverages"
)

cat("\nAssigned topic labels:\n")
print(topic_labels)

# --------------------------------------------------------------------------
# 5.4  ASSIGN DOMINANT TOPIC PER REVIEW
# --------------------------------------------------------------------------
cat("\nAssigning dominant topic to each review...\n")

doc_topics <- tidy(lda_model, matrix = "gamma") %>%
  group_by(document) %>%
  slice_max(gamma, n = 1) %>%
  ungroup() %>%
  mutate(
    doc_id        = as.integer(document),
    dominant_topic = topic,
    topic_label   = topic_labels[as.character(topic)]
  ) %>%
  select(doc_id, dominant_topic, topic_label)

amazon_lda <- amazon_lda %>%
  left_join(doc_topics, by = "doc_id")

cat("Dominant topic assigned to", sum(!is.na(amazon_lda$dominant_topic)), "reviews\n")

# --------------------------------------------------------------------------
# 5.5  TOPIC DISTRIBUTION
# --------------------------------------------------------------------------
cat("\n=== Topic Distribution ===\n")
amazon_lda %>%
  count(topic_label) %>%
  arrange(desc(n)) %>%
  print()

# --------------------------------------------------------------------------
# 5.6  VISUALISATIONS
# --------------------------------------------------------------------------
dir.create("figures", showWarnings = FALSE)

## Fig: Top terms per topic
p1 <- top_terms %>%
  mutate(topic_label = topic_labels[as.character(topic)],
         term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(x = beta, y = term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic_label, scales = "free_y", ncol = 4) +
  scale_y_reordered() +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "LDA Topic Modeling: Top 10 Keywords per Topic",
       x = "Word Probability (beta)", y = NULL) +
  theme_minimal(base_size = 10) +
  theme(plot.title = element_text(face = "bold"))

ggsave("figures/fig10_lda_top_terms.png", p1, width = 14, height = 8, dpi = 150)

## Fig: Topic frequency distribution
p2 <- amazon_lda %>%
  count(topic_label) %>%
  mutate(topic_label = fct_reorder(topic_label, n)) %>%
  ggplot(aes(x = n, y = topic_label, fill = topic_label)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = comma(n)), hjust = -0.1, size = 3.5) +
  scale_x_continuous(labels = comma, expand = expansion(mult = c(0, 0.15))) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Topic Distribution Across Reviews",
       x = "Number of Reviews", y = "Topic") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave("figures/fig11_topic_distribution.png", p2, width = 9, height = 6, dpi = 150)

cat("Saved: figures/fig10_lda_top_terms.png\n")
cat("Saved: figures/fig11_topic_distribution.png\n")

# --------------------------------------------------------------------------
# 5.7  SAVE
# --------------------------------------------------------------------------
saveRDS(lda_model,  "dataset/lda_model.rds")
saveRDS(amazon_lda, "dataset/amazon_stage5.rds")

cat("\n=== Stage 5 Complete ===\n")
cat("Saved: dataset/lda_model.rds\n")
cat("Saved: dataset/amazon_stage5.rds\n")
cat("Next: run 06_graph_analysis.R\n")