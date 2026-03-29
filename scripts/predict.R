# =============================================================================
# PREDICT: Is this review Fake or Genuine?
# Review Trust Analyzer | CSS-321 | Group 1
# =============================================================================
# Usage: change the review text below and run source("predict.R")
# =============================================================================

library(tidyverse)
library(tidytext)
library(SnowballC)
library(randomForest)

# -------------------------------------------------------
# CHANGE THIS TO ANY REVIEW YOU WANT TO TEST
review_text <- "This product is absolutely amazing, best thing I ever bought!!"
# -------------------------------------------------------

cat("=== Review Trust Analyzer — Prediction ===\n")
cat("Review:", review_text, "\n\n")

# Load saved assets
rf_model <- readRDS("trained_models/random_forest.rds")
vocab    <- readRDS("data/tfidf_vocabulary.rds")
fake_ref <- readRDS("data/fake_stage4.rds")

# --------------------------------------------------------------------------
# STEP 1: Get exact column names the model expects
# --------------------------------------------------------------------------
model_features <- rf_model$forest$ncat
model_colnames <- names(model_features)
cat("Model expects", length(model_colnames), "features\n")

# --------------------------------------------------------------------------
# STEP 2: Clean the review (same pipeline as training)
# --------------------------------------------------------------------------
stopwords_list <- stop_words$word

clean <- review_text %>%
  str_to_lower() %>%
  str_remove_all("https?://\\S+") %>%
  str_remove_all("[^a-z\\s]") %>%
  str_squish() %>%
  str_split("\\s+") %>%
  unlist()

clean <- clean[!(clean %in% stopwords_list)]
clean <- clean[nchar(clean) > 2]
clean <- wordStem(clean, language = "english")
clean_text <- paste(clean, collapse = " ")
cat("Cleaned text:", clean_text, "\n\n")

# --------------------------------------------------------------------------
# STEP 3: Build TF-IDF for top 100 vocab words
# --------------------------------------------------------------------------
top100 <- vocab[1:min(100, length(vocab))]

# Word frequencies from new review
new_word_tf <- table(clean) / max(length(clean), 1)

# IDF from training data
idf_ref <- fake_ref %>%
  mutate(doc_id = row_number()) %>%
  unnest_tokens(word, clean_text) %>%
  count(doc_id, word) %>%
  group_by(word) %>%
  summarise(idf = log(nrow(fake_ref) / n()), .groups = "drop")

# TF-IDF vector — one value per top100 word
tfidf_vec <- setNames(rep(0, length(top100)), top100)
for (w in names(new_word_tf)) {
  if (w %in% top100) {
    idf_val <- idf_ref %>% filter(word == w) %>% pull(idf)
    idf_val <- ifelse(length(idf_val) == 0, 0, idf_val[1])
    tfidf_vec[w] <- as.numeric(new_word_tf[w]) * idf_val
  }
}

# --------------------------------------------------------------------------
# STEP 4: Dense features
# --------------------------------------------------------------------------
review_length  <- str_count(review_text, "\\S+")
exclaim_count  <- str_count(review_text, "!")
caps_ratio     <- str_count(review_text, "[A-Z]") / max(nchar(review_text), 1)
avg_word_len   <- ifelse(length(clean) > 0,
                         nchar(paste(clean, collapse = "")) / length(clean), 0)
sentiment_score <- 0
rating_mismatch <- 0

dense_vec <- c(
  review_length   = review_length,
  exclaim_count   = exclaim_count,
  caps_ratio      = caps_ratio,
  avg_word_len    = avg_word_len,
  sentiment_score = sentiment_score,
  rating_mismatch = rating_mismatch
)

# --------------------------------------------------------------------------
# STEP 5: Build feature row with EXACT columns the model expects
# --------------------------------------------------------------------------
all_features <- c(tfidf_vec, dense_vec)

# Create a row with all model columns, defaulting missing ones to 0
feature_row <- setNames(rep(0, length(model_colnames)), model_colnames)
matched <- intersect(names(all_features), model_colnames)
feature_row[matched] <- all_features[matched]

feature_df <- as.data.frame(t(feature_row))

# --------------------------------------------------------------------------
# STEP 6: Predict
# --------------------------------------------------------------------------
prediction  <- predict(rf_model, feature_df)
probability <- predict(rf_model, feature_df, type = "prob")

cat("=== RESULT ===\n")
cat("Verdict:     ", as.character(prediction), "\n")
cat("Fake prob:   ", round(probability[1, "Fake"]    * 100, 1), "%\n")
cat("Genuine prob:", round(probability[1, "Genuine"] * 100, 1), "%\n")

if (as.character(prediction) == "Fake") {
  cat("\n[!] This review is likely FAKE\n")
} else {
  cat("\n[✓] This review appears GENUINE\n")
}