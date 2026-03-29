# =============================================================================
# STAGE 7: Product-Level Anomaly Detection using Isolation Forest
# Review Trust Analyzer | CSS-321 | Group 1
# =============================================================================
# OBJECTIVE: Detect products with statistically abnormal review patterns
# — the signature of coordinated fake review floods.
# =============================================================================

library(tidyverse)
library(isotree)
library(scales)

cat("Loading Stage 6 data...\n")
amazon <- readRDS("data/amazon_stage6.rds")
cat("Amazon:", nrow(amazon), "rows\n")

# --------------------------------------------------------------------------
# 7.1  COMPUTE PRODUCT-LEVEL AGGREGATE FEATURES
# --------------------------------------------------------------------------
cat("\nAggregating features at product level...\n")

product_features <- amazon %>%
  group_by(asin) %>%
  summarise(
    review_count     = n(),
    avg_rating       = mean(overall, na.rm = TRUE),
    rating_variance  = var(overall, na.rm = TRUE),
    five_star_ratio  = mean(overall == 5, na.rm = TRUE),
    sentiment_std    = sd(sentiment_score, na.rm = TRUE),
    avg_sentiment    = mean(sentiment_score, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  # Only keep products with at least 5 reviews (insufficient signal otherwise)
  filter(review_count >= 5) %>%
  mutate(
    rating_variance = replace_na(rating_variance, 0),
    sentiment_std   = replace_na(sentiment_std, 0)
  )

cat("Products with >= 5 reviews:", nrow(product_features), "\n")
cat("\n=== Product Feature Summary ===\n")
print(summary(product_features[, c("review_count","avg_rating",
                                    "five_star_ratio","sentiment_std")]))

# --------------------------------------------------------------------------
# 7.2  FIT ISOLATION FOREST
# --------------------------------------------------------------------------
cat("\nFitting Isolation Forest (contamination = 10%)...\n")

feature_matrix <- product_features %>%
  select(review_count, avg_rating, rating_variance,
         five_star_ratio, sentiment_std) %>%
  as.data.frame()

if (nrow(feature_matrix) == 0) {
  stop("No products available for anomaly detection after filtering (review_count >= 5).")
}

set.seed(42)

iso_model <- isolation.forest(
  feature_matrix,
  ntrees       = 100,
  sample_size  = min(256, nrow(feature_matrix))
)

# Anomaly scores — higher = more anomalous
product_features <- product_features %>%
  mutate(
    anomaly_score       = predict(iso_model, feature_matrix),
    # Top 10% most anomalous = flagged
    product_anomaly_flag = ifelse(anomaly_score >= quantile(anomaly_score, 0.90), 1L, 0L)
  )

cat("Anomalous products flagged:", sum(product_features$product_anomaly_flag), "\n")
cat("Normal products:           ", sum(product_features$product_anomaly_flag == 0), "\n")

# --------------------------------------------------------------------------
# 7.3  INSPECT TOP ANOMALOUS PRODUCTS
# --------------------------------------------------------------------------
cat("\n=== Top 10 Most Anomalous Products ===\n")
product_features %>%
  filter(product_anomaly_flag == 1) %>%
  arrange(desc(anomaly_score)) %>%
  select(asin, review_count, avg_rating, five_star_ratio,
         sentiment_std, anomaly_score) %>%
  slice_head(n = 10) %>%
  print()

# --------------------------------------------------------------------------
# 7.4  MAP ANOMALY FLAG BACK TO INDIVIDUAL REVIEWS
# --------------------------------------------------------------------------
cat("\nMapping anomaly flags back to individual reviews...\n")

amazon <- amazon %>%
  left_join(
    product_features %>% select(asin, anomaly_score, product_anomaly_flag),
    by = "asin"
  ) %>%
  mutate(
    anomaly_score        = replace_na(anomaly_score, 0),
    product_anomaly_flag = replace_na(product_anomaly_flag, 0L)
  )

cat("Reviews flagged as from anomalous products:",
    sum(amazon$product_anomaly_flag), "\n")
cat("Percentage:", round(mean(amazon$product_anomaly_flag) * 100, 1), "%\n")

# --------------------------------------------------------------------------
# 7.5  VISUALISATIONS
# --------------------------------------------------------------------------
dir.create("figures", showWarnings = FALSE)

## Fig: Anomaly scatter — review count vs sentiment std
p1 <- product_features %>%
  mutate(flag = ifelse(product_anomaly_flag == 1, "Anomalous", "Normal")) %>%
  ggplot(aes(x = review_count, y = sentiment_std, color = flag)) +
  geom_point(alpha = 0.5, size = 1.5) +
  scale_color_manual(values = c("Normal" = "#2ca25f", "Anomalous" = "#de2d26")) +
  scale_x_log10(labels = comma) +
  labs(title = "Product Anomaly Detection: Review Count vs Sentiment Variance",
       subtitle = "Red = flagged as anomalous by Isolation Forest",
       x = "Review Count (log scale)", y = "Sentiment Std Dev", color = "Status") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"), legend.position = "top")

ggsave("results/figures/fig15_anomaly_scatter.png", p1, width = 8, height = 6, dpi = 150)

## Fig: Anomaly score distribution
p2 <- product_features %>%
  ggplot(aes(x = anomaly_score)) +
  geom_histogram(bins = 40, fill = "#2c7bb6", color = "white", alpha = 0.85) +
  geom_vline(xintercept = quantile(product_features$anomaly_score, 0.90),
             color = "red", linetype = "dashed", linewidth = 1) +
  annotate("text", x = quantile(product_features$anomaly_score, 0.90) + 0.01,
           y = Inf, label = "90th percentile\n(anomaly threshold)",
           vjust = 2, hjust = 0, color = "red", size = 3.5) +
  scale_y_continuous(labels = comma) +
  labs(title = "Isolation Forest: Anomaly Score Distribution",
       x = "Anomaly Score", y = "Number of Products") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave("results/figures/fig16_anomaly_score_dist.png", p2, width = 7, height = 5, dpi = 150)

cat("Saved: results/figures/fig15_anomaly_scatter.png\n")
cat("Saved: results/figures/fig16_anomaly_score_dist.png\n")

# --------------------------------------------------------------------------
# 7.6  SAVE
# --------------------------------------------------------------------------
dir.create("trained_models", showWarnings = FALSE)

saveRDS(amazon,           "data/amazon_stage7.rds")
saveRDS(product_features, "data/product_features.rds")
saveRDS(iso_model,        "trained_models/isolation_forest.rds")

cat("\n=== Stage 7 Complete ===\n")
cat("Saved: data/amazon_stage7.rds\n")
cat("Saved: data/product_features.rds\n")
cat("Saved: trained_models/isolation_forest.rds\n")
cat("Next: run 08_classification.R\n")