# =============================================================================
# STAGE 1: Dataset Loading & Exploratory Data Analysis
# Review Trust Analyzer | CSS-321 | Group 1
# =============================================================================
# OBJECTIVE: Understand data shape, check for nulls, visualize class balance,
# and compute basic statistics before writing any pipeline code.
# =============================================================================

library(tidyverse)
library(lubridate)
library(scales)
library(ggplot2)

# --------------------------------------------------------------------------
# 1.1  LOAD DATASETS
# --------------------------------------------------------------------------
# Make sure you have placed the downloaded CSVs in the data/ folder:
#   data/amazon_reviews.csv   ← from kaggle.com/datasets/snap/amazon-fine-food-reviews
#   data/fake_reviews.csv     ← from kaggle.com/datasets/mexwell/fake-reviews-dataset

cat("=== Loading datasets ===\n")

amazon <- read_csv("data/amazon_reviews.csv", show_col_types = FALSE)
fake   <- read_csv("data/fake_reviews.csv",   show_col_types = FALSE)

# --------------------------------------------------------------------------
# 1.2  STANDARDISE COLUMN NAMES
# --------------------------------------------------------------------------
# Amazon dataset column mapping to match project spec:
#   UserId      → reviewerID
#   ProductId   → asin
#   Text        → reviewText
#   Score       → overall
#   Time        → unixReviewTime  (Unix timestamp already)
#   HelpfulnessDenominator → helpful proxy (no 'verified' in this dataset)

amazon <- amazon %>%
  rename(
    reviewerID      = UserId,
    asin            = ProductId,
    reviewText      = Text,
    overall         = Score,
    unixReviewTime  = Time,
    summary_text    = Summary
  ) %>%
  mutate(
    # Create a 'verified' proxy: reviews where helpfulness votes > 0
    # treat as more credible; 0 votes = potentially suspicious
    verified = ifelse(HelpfulnessDenominator > 0, TRUE, FALSE),
    # Convert Unix timestamp to date
    review_date = as_datetime(unixReviewTime)
  )

# Fake dataset column mapping:
#   label: "CG" = Computer-Generated (fake), "OR" = Original (genuine)
#   text_  → reviewText
#   rating → overall
fake <- fake %>%
  rename(
    reviewText = text_,
    overall    = rating
  ) %>%
  mutate(
    # Recode to binary numeric: 1 = fake, 0 = genuine
    fake_label = ifelse(label == "CG", 1L, 0L),
    label_name = ifelse(label == "CG", "Fake", "Genuine")
  )

# --------------------------------------------------------------------------
# 1.3  BASIC DATA INSPECTION
# --------------------------------------------------------------------------
cat("\n=== AMAZON DATASET ===\n")
cat("Shape:", nrow(amazon), "rows x", ncol(amazon), "cols\n")
cat("Columns:", paste(names(amazon), collapse = ", "), "\n")
cat("\nNull counts:\n")
print(colSums(is.na(amazon)))

cat("\n=== FAKE REVIEWS DATASET ===\n")
cat("Shape:", nrow(fake), "rows x", ncol(fake), "cols\n")
cat("Columns:", paste(names(fake), collapse = ", "), "\n")
cat("\nNull counts:\n")
print(colSums(is.na(fake)))

# --------------------------------------------------------------------------
# 1.4  SAMPLE AMAZON DATASET TO 80K ROWS
# --------------------------------------------------------------------------
cat("\n=== Sampling Amazon dataset to 80,000 rows ===\n")
set.seed(42)
amazon <- amazon %>%
  filter(!is.na(reviewText), nchar(reviewText) > 10) %>%
  sample_n(min(80000, n()))

cat("Amazon after sampling:", nrow(amazon), "rows\n")

# Remove rows with missing review text in fake dataset
fake <- fake %>%
  filter(!is.na(reviewText), nchar(reviewText) > 5)
cat("Fake reviews after cleaning:", nrow(fake), "rows\n")

# --------------------------------------------------------------------------
# 1.5  REVIEW LENGTH FEATURE (pre-computation)
# --------------------------------------------------------------------------
amazon <- amazon %>%
  mutate(review_length = str_count(reviewText, "\\S+"))

fake <- fake %>%
  mutate(review_length = str_count(reviewText, "\\S+"))

# --------------------------------------------------------------------------
# 1.6  SUMMARY STATISTICS
# --------------------------------------------------------------------------
cat("\n=== Amazon: Star Rating Distribution ===\n")
print(table(amazon$overall))

cat("\n=== Fake Reviews: Class Balance ===\n")
print(table(fake$label_name))

cat("\n=== Amazon: Verified vs Unverified ===\n")
print(table(amazon$verified))

cat("\n=== Amazon: Review Length Summary ===\n")
print(summary(amazon$review_length))

# --------------------------------------------------------------------------
# 1.7  VISUALIZATIONS
# --------------------------------------------------------------------------
dir.create("figures", showWarnings = FALSE)
theme_set(theme_minimal(base_size = 13))

## --- Fig A: Amazon Star Rating Distribution ---
p1 <- ggplot(amazon, aes(x = factor(overall))) +
  geom_bar(aes(fill = factor(overall)), show.legend = FALSE) +
  geom_text(stat = "count", aes(label = comma(..count..)),
            vjust = -0.4, size = 3.5) +
  scale_fill_brewer(palette = "Blues") +
  scale_y_continuous(labels = comma) +
  labs(title = "Amazon Reviews: Star Rating Distribution",
       subtitle = "Sampled 80K reviews",
       x = "Star Rating", y = "Count") +
  theme(plot.title = element_text(face = "bold"))

ggsave("results/figures/eda_01_rating_distribution.png", p1,
       width = 7, height = 5, dpi = 150)
cat("Saved: results/figures/eda_01_rating_distribution.png\n")

## --- Fig B: Review Length Histogram ---
p2 <- ggplot(amazon %>% filter(review_length < 400),
             aes(x = review_length)) +
  geom_histogram(bins = 50, fill = "#2c7bb6", color = "white", alpha = 0.85) +
  scale_y_continuous(labels = comma) +
  labs(title = "Amazon Reviews: Review Length Distribution",
       subtitle = "Word count per review (truncated at 400 words for clarity)",
       x = "Number of Words", y = "Count") +
  theme(plot.title = element_text(face = "bold"))

ggsave("results/figures/eda_02_review_length.png", p2,
       width = 7, height = 5, dpi = 150)
cat("Saved: results/figures/eda_02_review_length.png\n")

## --- Fig C: Fake Dataset Class Balance ---
fake_counts <- fake %>%
  count(label_name) %>%
  mutate(pct = n / sum(n),
         label_pct = paste0(label_name, "\n", percent(pct, 0.1)))

p3 <- ggplot(fake_counts, aes(x = label_name, y = n, fill = label_name)) +
  geom_col(width = 0.5, show.legend = FALSE) +
  geom_text(aes(label = paste0(comma(n), "\n(", percent(pct, 0.1), ")")),
            vjust = -0.3, size = 4) +
  scale_fill_manual(values = c("Genuine" = "#2ca25f", "Fake" = "#de2d26")) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.12))) +
  labs(title = "Fake Reviews Dataset: Class Balance",
       subtitle = "CG = Computer Generated (Fake) | OR = Original (Genuine)",
       x = "Review Type", y = "Count") +
  theme(plot.title = element_text(face = "bold"))

ggsave("results/figures/eda_03_class_balance.png", p3,
       width = 6, height = 5, dpi = 150)
cat("Saved: results/figures/eda_03_class_balance.png\n")

## --- Fig D: Verified vs Unverified (Amazon) ---
p4 <- amazon %>%
  count(verified) %>%
  mutate(label = ifelse(verified, "Verified", "Unverified"),
         pct   = n / sum(n)) %>%
  ggplot(aes(x = label, y = n, fill = label)) +
  geom_col(width = 0.5, show.legend = FALSE) +
  geom_text(aes(label = paste0(comma(n), "\n(", percent(pct, 0.1), ")")),
            vjust = -0.3, size = 4) +
  scale_fill_manual(values = c("Verified" = "#4dac26", "Unverified" = "#d01c8b")) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.12))) +
  labs(title = "Amazon Reviews: Verified Purchase Status",
       x = "Status", y = "Count") +
  theme(plot.title = element_text(face = "bold"))

ggsave("results/figures/eda_04_verified_status.png", p4,
       width = 6, height = 5, dpi = 150)
cat("Saved: results/figures/eda_04_verified_status.png\n")

## --- Fig E: Review Length by Fake vs Genuine ---
p5 <- ggplot(fake %>% filter(review_length < 300),
             aes(x = review_length, fill = label_name)) +
  geom_density(alpha = 0.6) +
  scale_fill_manual(values = c("Genuine" = "#2ca25f", "Fake" = "#de2d26")) +
  labs(title = "Review Length: Fake vs Genuine",
       subtitle = "Fake reviews tend to be shorter",
       x = "Number of Words", y = "Density", fill = "Type") +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "top")

ggsave("results/figures/eda_05_length_by_label.png", p5,
       width = 7, height = 5, dpi = 150)
cat("Saved: results/figures/eda_05_length_by_label.png\n")

# --------------------------------------------------------------------------
# 1.8  SAVE CLEANED DATASETS FOR NEXT STAGE
# --------------------------------------------------------------------------
saveRDS(amazon, "data/amazon_clean_stage1.rds")
saveRDS(fake,   "data/fake_clean_stage1.rds")

cat("\n=== Stage 1 Complete ===\n")
cat("Outputs:\n")
cat("  data/amazon_clean_stage1.rds\n")
cat("  data/fake_clean_stage1.rds\n")
cat("  results/figures/eda_01 through eda_05\n")
cat("\nNext step: Run R/02_preprocess.R\n")