# =============================================================================
# STAGE 8: Fake Review Classification
# Review Trust Analyzer | CSS-321 | Group 1
# =============================================================================
# OBJECTIVE: Train and compare three classifiers on the unified feature matrix.
# Report full evaluation metrics. Save trained models.
# =============================================================================

library(tidyverse)
library(tidytext)
library(Matrix)
library(caret)
library(randomForest)
library(e1071)
library(scales)

cat("Loading data...\n")
fake   <- readRDS("data/fake_stage4.rds")
vocab  <- readRDS("data/tfidf_vocabulary.rds")
cat("Fake reviews:", nrow(fake), "rows\n")

# --------------------------------------------------------------------------
# 8.1  BUILD FEATURE MATRIX
# --------------------------------------------------------------------------
cat("\nBuilding unified feature matrix...\n")

# --- TF-IDF features (sparse) ---
cat("  Computing TF-IDF...\n")
tfidf_long <- fake %>%
  mutate(doc_id = row_number()) %>%
  select(doc_id, clean_text) %>%
  unnest_tokens(word, clean_text) %>%
  filter(word %in% vocab) %>%
  count(doc_id, word) %>%
  bind_tf_idf(word, doc_id, n)

# Build sparse matrix
all_docs  <- 1:nrow(fake)
all_words <- vocab

tfidf_wide <- tfidf_long %>%
  select(doc_id, word, tf_idf) %>%
  pivot_wider(names_from = word, values_from = tf_idf,
              values_fill = 0, id_cols = doc_id)

# Ensure all docs present
missing_docs <- setdiff(all_docs, tfidf_wide$doc_id)
if (length(missing_docs) > 0) {
  empty_rows <- matrix(0, nrow = length(missing_docs),
                       ncol = ncol(tfidf_wide) - 1)
  empty_df   <- as.data.frame(empty_rows)
  names(empty_df) <- names(tfidf_wide)[-1]
  empty_df$doc_id <- missing_docs
  tfidf_wide <- bind_rows(tfidf_wide, empty_df) %>% arrange(doc_id)
}

tfidf_matrix <- tfidf_wide %>% select(-doc_id) %>% as.matrix()
cat("  TF-IDF matrix:", nrow(tfidf_matrix), "x", ncol(tfidf_matrix), "\n")

# --- Handcrafted + sentiment features (dense) ---
dense_features <- fake %>%
  select(review_length, exclaim_count, caps_ratio,
         avg_word_len, sentiment_score, rating_mismatch) %>%
  mutate(across(everything(), ~ replace_na(., 0))) %>%
  as.matrix()

cat("  Dense features:", ncol(dense_features), "columns\n")

# --- Combine into final feature matrix ---
X <- cbind(tfidf_matrix, dense_features)
y <- factor(fake$fake_label, levels = c(0, 1),
            labels = c("Genuine", "Fake"))

cat("Final feature matrix:", nrow(X), "rows x", ncol(X), "cols\n")
cat("Class distribution:\n")
print(table(y))

# --------------------------------------------------------------------------
# 8.2  TRAIN / TEST SPLIT (80/20 stratified)
# --------------------------------------------------------------------------
set.seed(42)
train_idx <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_idx, ];  y_train <- y[train_idx]
X_test  <- X[-train_idx, ]; y_test  <- y[-train_idx]

cat("\nTrain size:", length(y_train), "| Test size:", length(y_test), "\n")

# --------------------------------------------------------------------------
# 8.3  MODEL 1 — NAIVE BAYES (on dense features only, needs non-negative)
# --------------------------------------------------------------------------
cat("\n--- Training Naive Bayes ---\n")
nb_model <- naiveBayes(x = dense_features[train_idx, ],
                       y = y_train)
nb_pred  <- predict(nb_model, dense_features[-train_idx, ])

cat("Naive Bayes Results:\n")
nb_cm <- confusionMatrix(nb_pred, y_test, positive = "Fake")
print(nb_cm$overall[c("Accuracy")])
print(nb_cm$byClass[c("Precision","Recall","F1")])

# --------------------------------------------------------------------------
# 8.4  MODEL 2 — LOGISTIC REGRESSION
# --------------------------------------------------------------------------
cat("\n--- Training Logistic Regression ---\n")

# Use dense features for LR (full TF-IDF too large for glm)
lr_model <- glm(y_train ~ .,
                data = as.data.frame(dense_features[train_idx, ]),
                family = binomial(link = "logit"))

lr_prob <- predict(lr_model,
                   newdata = as.data.frame(dense_features[-train_idx, ]),
                   type = "response")
lr_pred <- factor(ifelse(lr_prob > 0.5, "Fake", "Genuine"),
                  levels = c("Genuine", "Fake"))

cat("Logistic Regression Results:\n")
lr_cm <- confusionMatrix(lr_pred, y_test, positive = "Fake")
print(lr_cm$overall[c("Accuracy")])
print(lr_cm$byClass[c("Precision","Recall","F1")])

# --------------------------------------------------------------------------
# 8.5  MODEL 3 — RANDOM FOREST
# --------------------------------------------------------------------------
cat("\n--- Training Random Forest (this takes 3-5 mins) ---\n")

# Use dense features + sample of TF-IDF top 100 terms for RF
top100 <- colnames(tfidf_matrix)[1:min(100, ncol(tfidf_matrix))]
X_rf_train <- cbind(tfidf_matrix[train_idx, top100], dense_features[train_idx, ])
X_rf_test  <- cbind(tfidf_matrix[-train_idx, top100], dense_features[-train_idx, ])

rf_model <- randomForest(
  x          = X_rf_train,
  y          = y_train,
  ntree      = 200,
  importance = TRUE,
  random.seed = 42
)

rf_pred <- predict(rf_model, X_rf_test)

cat("Random Forest Results:\n")
rf_cm <- confusionMatrix(rf_pred, y_test, positive = "Fake")
print(rf_cm$overall[c("Accuracy")])
print(rf_cm$byClass[c("Precision","Recall","F1")])

# --------------------------------------------------------------------------
# 8.6  MODEL COMPARISON TABLE
# --------------------------------------------------------------------------
cat("\n=== MODEL COMPARISON ===\n")
comparison <- tibble(
  Model     = c("Naive Bayes", "Logistic Regression", "Random Forest"),
  Accuracy  = round(c(nb_cm$overall["Accuracy"],
                      lr_cm$overall["Accuracy"],
                      rf_cm$overall["Accuracy"]), 3),
  Precision = round(c(nb_cm$byClass["Precision"],
                      lr_cm$byClass["Precision"],
                      rf_cm$byClass["Precision"]), 3),
  Recall    = round(c(nb_cm$byClass["Recall"],
                      lr_cm$byClass["Recall"],
                      rf_cm$byClass["Recall"]), 3),
  F1        = round(c(nb_cm$byClass["F1"],
                      lr_cm$byClass["F1"],
                      rf_cm$byClass["F1"]), 3)
)
print(comparison)

# --------------------------------------------------------------------------
# 8.7  VISUALISATIONS
# --------------------------------------------------------------------------
dir.create("figures", showWarnings = FALSE)

## Confusion matrix for best model (Random Forest)
cm_df <- as.data.frame(rf_cm$table) %>%
  group_by(Reference) %>%
  mutate(pct = Freq / sum(Freq))

p1 <- ggplot(cm_df, aes(x = Reference, y = Prediction, fill = pct)) +
  geom_tile(color = "white") +
  geom_text(aes(label = paste0(Freq, "\n(", percent(pct, 1), ")")),
            size = 4, fontface = "bold", color = "white") +
  scale_fill_gradient(low = "#deebf7", high = "#08519c", labels = percent) +
  labs(title = "Random Forest: Confusion Matrix",
       x = "Actual", y = "Predicted", fill = "Rate") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave("results/figures/fig17_confusion_matrix.png", p1, width = 6, height = 5, dpi = 150)

## Feature importance
imp_df <- importance(rf_model) %>%
  as.data.frame() %>%
  rownames_to_column("feature") %>%
  arrange(desc(MeanDecreaseGini)) %>%
  slice_head(n = 20)

p2 <- ggplot(imp_df, aes(x = MeanDecreaseGini,
                          y = reorder(feature, MeanDecreaseGini))) +
  geom_col(fill = "#2c7bb6", alpha = 0.85) +
  labs(title = "Random Forest: Top 20 Feature Importances",
       x = "Mean Decrease in Gini", y = "Feature") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))

ggsave("results/figures/fig18_feature_importance.png", p2, width = 8, height = 6, dpi = 150)

## Model comparison bar chart
p3 <- comparison %>%
  pivot_longer(cols = c(Precision, Recall, F1),
               names_to = "Metric", values_to = "Score") %>%
  ggplot(aes(x = Model, y = Score, fill = Metric)) +
  geom_col(position = "dodge", alpha = 0.85) +
  geom_text(aes(label = round(Score, 2)),
            position = position_dodge(width = 0.9),
            vjust = -0.3, size = 3) +
  scale_fill_brewer(palette = "Set1") +
  scale_y_continuous(limits = c(0, 1.1), labels = percent) +
  labs(title = "Model Comparison: Precision, Recall & F1",
       x = NULL, y = "Score", fill = "Metric") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "top")

ggsave("results/figures/fig19_model_comparison.png", p3, width = 8, height = 5, dpi = 150)

cat("Saved: results/figures/fig17, fig18, fig19\n")

# --------------------------------------------------------------------------
# 8.8  SAVE MODELS
# --------------------------------------------------------------------------
dir.create("trained_models", showWarnings = FALSE)
saveRDS(nb_model, "trained_models/naive_bayes.rds")
saveRDS(lr_model, "trained_models/logistic_regression.rds")
saveRDS(rf_model, "trained_models/random_forest.rds")

cat("\n=== Stage 8 Complete ===\n")
cat("Saved: trained_models/naive_bayes.rds\n")
cat("Saved: trained_models/logistic_regression.rds\n")
cat("Saved: trained_models/random_forest.rds\n")
cat("Saved: results/figures/fig17, fig18, fig19\n")
cat("\n ALL STAGES COMPLETE — Project pipeline finished!\n")