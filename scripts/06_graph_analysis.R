# =============================================================================
# STAGE 6: Reviewer Behavioral Graph Analysis
# Review Trust Analyzer | CSS-321 | Group 1
# =============================================================================
# OBJECTIVE: Build a reviewer graph to detect coordinated fake review
# patterns invisible to text classifiers.
# =============================================================================

library(tidyverse)
library(igraph)
library(scales)

cat("Loading Stage 4 data...\n")
amazon <- readRDS("data/amazon_stage4.rds")
cat("Amazon:", nrow(amazon), "rows\n")

# --------------------------------------------------------------------------
# 6.1  COMPUTE PER-REVIEWER FEATURES
# --------------------------------------------------------------------------
cat("\nComputing per-reviewer behavioral features...\n")

reviewer_features <- amazon %>%
  arrange(reviewerID, unixReviewTime) %>%
  group_by(reviewerID) %>%
  summarise(
    total_reviews    = n(),
    five_star_ratio  = mean(overall == 5),
    avg_rating       = mean(overall),
    review_velocity  = n() / max(1, as.numeric(
                         difftime(max(review_date), min(review_date), units = "days")
                       )),
    # burst_score: std of time gaps in hours — low = robotic regularity
    burst_score      = ifelse(
      n() > 1,
      sd(diff(sort(unixReviewTime)) / 3600),
      0
    ),
    .groups = "drop"
  ) %>%
  mutate(
    # Replace NA burst scores (single-review accounts)
    burst_score = replace_na(burst_score, 0)
  )

cat("Reviewer features computed for", nrow(reviewer_features), "reviewers\n")
cat("\n=== Reviewer Feature Summary ===\n")
print(summary(reviewer_features[, c("total_reviews","five_star_ratio",
                                     "review_velocity","burst_score")]))

# --------------------------------------------------------------------------
# 6.2  FIND SUSPICIOUS CO-REVIEW PAIRS
# --------------------------------------------------------------------------
cat("\nFinding suspicious co-review pairs (same product within 48 hours)...\n")

# For efficiency, only look at products with multiple reviews
multi_reviewed <- amazon %>%
  group_by(asin) %>%
  filter(n() >= 2) %>%
  ungroup()

# Self-join on same product, different reviewer, within 48 hours
co_reviews <- multi_reviewed %>%
  select(asin, reviewerID, unixReviewTime) %>%
  inner_join(
    multi_reviewed %>% select(asin, reviewerID2 = reviewerID,
                               time2 = unixReviewTime),
    by = "asin"
  ) %>%
  filter(
    reviewerID != reviewerID2,
    abs(unixReviewTime - time2) <= 48 * 3600  # within 48 hours
  ) %>%
  # Keep unique pairs only (A-B same as B-A)
  filter(reviewerID < reviewerID2) %>%
  group_by(reviewerID, reviewerID2) %>%
  summarise(shared_windows = n(), .groups = "drop")

cat("Co-review pairs found:", nrow(co_reviews), "\n")

# --------------------------------------------------------------------------
# 6.3  BUILD REVIEWER-REVIEWER GRAPH
# --------------------------------------------------------------------------
cat("\nBuilding reviewer-reviewer graph...\n")

if (nrow(co_reviews) > 0) {
  g <- graph_from_data_frame(
    co_reviews %>% rename(from = reviewerID, to = reviewerID2,
                          weight = shared_windows),
    directed = FALSE
  )

  # Find connected components
  components   <- components(g)
  component_df <- data.frame(
    reviewerID    = names(components$membership),
    component_id  = as.integer(components$membership)
  )

  # Flag reviewers in suspicious clusters (component size > 3)
  component_sizes <- component_df %>%
    group_by(component_id) %>%
    mutate(component_size = n()) %>%
    ungroup()

  component_df <- component_df %>%
    left_join(
      component_sizes %>% select(reviewerID, component_size),
      by = "reviewerID"
    ) %>%
    mutate(is_cluster_member = ifelse(component_size > 3, 1L, 0L))

  cat("Total reviewers in graph:", vcount(g), "\n")
  cat("Connected components:", components$no, "\n")
  cat("Suspicious cluster members (size > 3):",
      sum(component_df$is_cluster_member), "\n")

  # Merge cluster flag into reviewer features
  reviewer_features <- reviewer_features %>%
    left_join(component_df %>% select(reviewerID, is_cluster_member,
                                       component_size),
              by = "reviewerID") %>%
    mutate(
      is_cluster_member = replace_na(is_cluster_member, 0L),
      component_size    = replace_na(component_size, 1L)
    )

} else {
  cat("No co-review pairs found — assigning default cluster values\n")
  reviewer_features <- reviewer_features %>%
    mutate(is_cluster_member = 0L, component_size = 1L)
  g <- make_empty_graph()
}

# --------------------------------------------------------------------------
# 6.4  MERGE REVIEWER FEATURES BACK INTO MAIN DATAFRAME
# --------------------------------------------------------------------------
cat("\nMerging reviewer features into main dataframe...\n")

amazon <- amazon %>%
  left_join(
    reviewer_features %>% select(reviewerID, five_star_ratio, review_velocity,
                                  burst_score, is_cluster_member),
    by = "reviewerID"
  ) %>%
  mutate(
    five_star_ratio   = replace_na(five_star_ratio, 0),
    review_velocity   = replace_na(review_velocity, 0),
    burst_score       = replace_na(burst_score, 0),
    is_cluster_member = replace_na(is_cluster_member, 0L)
  )

cat("Reviewer features merged.\n")

# --------------------------------------------------------------------------
# 6.5  SUSPICIOUS REVIEWER STATS
# --------------------------------------------------------------------------
cat("\n=== Top 10 Most Suspicious Reviewers (high velocity + high 5-star) ===\n")
reviewer_features %>%
  filter(total_reviews >= 3) %>%
  mutate(suspicion_score = five_star_ratio * review_velocity) %>%
  arrange(desc(suspicion_score)) %>%
  select(reviewerID, total_reviews, five_star_ratio,
         review_velocity, burst_score, is_cluster_member) %>%
  slice_head(n = 10) %>%
  print()

# --------------------------------------------------------------------------
# 6.6  VISUALISATIONS
# --------------------------------------------------------------------------
dir.create("figures", showWarnings = FALSE)

## Fig: Five-star ratio distribution
p1 <- reviewer_features %>%
  ggplot(aes(x = five_star_ratio)) +
  geom_histogram(bins = 20, fill = "#2c7bb6", color = "white", alpha = 0.85) +
  scale_y_continuous(labels = comma) +
  labs(title = "Reviewer Five-Star Ratio Distribution",
       subtitle = "Reviewers with ratio = 1.0 only ever give 5 stars — suspicious",
       x = "Five-Star Ratio", y = "Number of Reviewers") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave("results/figures/fig12_five_star_ratio.png", p1, width = 7, height = 5, dpi = 150)

## Fig: Review velocity distribution
p2 <- reviewer_features %>%
  filter(review_velocity < 5) %>%
  ggplot(aes(x = review_velocity)) +
  geom_histogram(bins = 30, fill = "#de2d26", color = "white", alpha = 0.85) +
  scale_y_continuous(labels = comma) +
  labs(title = "Review Velocity Distribution",
       subtitle = "Reviews per day per reviewer",
       x = "Reviews per Day", y = "Number of Reviewers") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave("results/figures/fig13_review_velocity.png", p2, width = 7, height = 5, dpi = 150)

## Fig: Graph plot of largest suspicious cluster
if (vcount(g) > 0 && ecount(g) > 0) {
  # Find the largest component
  comp      <- components(g)
  largest   <- which.max(comp$csize)
  subg      <- induced_subgraph(g, which(comp$membership == largest))

  png("results/figures/fig14_reviewer_graph.png", width = 800, height = 800, res = 150)
  plot(subg,
       vertex.color = "#de2d26",
       vertex.size  = 6,
       vertex.label = NA,
       edge.color   = "#888888",
       edge.width   = E(subg)$weight,
       layout       = layout_with_fr(subg),
       main         = "Largest Suspicious Co-Review Cluster")
  dev.off()
  cat("Saved: results/figures/fig14_reviewer_graph.png\n")
} else {
  cat("Graph too sparse for visualization — skipping graph plot\n")
}

cat("Saved: results/figures/fig12, fig13\n")

# --------------------------------------------------------------------------
# 6.7  SAVE
# --------------------------------------------------------------------------
saveRDS(amazon,           "data/amazon_stage6.rds")
saveRDS(reviewer_features,"data/reviewer_features.rds")

cat("\n=== Stage 6 Complete ===\n")
cat("Saved: data/amazon_stage6.rds\n")
cat("Saved: data/reviewer_features.rds\n")
cat("Next: run 07_anomaly_detection.R\n")