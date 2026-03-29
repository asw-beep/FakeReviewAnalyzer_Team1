# =============================================================================
# PLUMBER API: Review Trust Analyzer
# Serves predictions from the Random Forest model via HTTP
# =============================================================================
# Run: Rscript -e "plumber::plumb('api.R')\$run(port=8787, host='0.0.0.0')"
# =============================================================================

library(plumber)
library(tidyverse)
library(tidytext)
library(SnowballC)
library(randomForest)

# ── Load assets once at startup ──────────────────────────────────────────────
cat("Loading model and vocabulary...\n")
rf_model <- readRDS("trained_models/random_forest.rds")
vocab    <- readRDS("data/tfidf_vocabulary.rds")
fake_ref <- readRDS("data/fake_stage4.rds")

model_features <- rf_model$forest$ncat
model_colnames <- names(model_features)
stopwords_list <- stop_words$word

# Pre-compute IDF from training data
idf_ref <- fake_ref %>%
  mutate(doc_id = row_number()) %>%
  unnest_tokens(word, clean_text) %>%
  count(doc_id, word) %>%
  group_by(word) %>%
  summarise(idf = log(nrow(fake_ref) / n()), .groups = "drop")

cat("API ready. Model expects", length(model_colnames), "features\n")

#* @filter cors
function(req, res) {
  res$setHeader("Access-Control-Allow-Origin", "*")
  res$setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
  res$setHeader("Access-Control-Allow-Headers", "Content-Type")
  if (req$REQUEST_METHOD == "OPTIONS") {
    res$status <- 200
    return(list())
  }
  plumber::forward()
}

#* Predict whether a review is Fake or Genuine
#* @param review_text The review text to analyse
#* @post /predict
function(review_text = "") {
  if (nchar(trimws(review_text)) == 0) {
    return(list(error = "Please provide review_text"))
  }

  # ── Clean ──────────────────────────────────────────────────────────────────
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

  # ── TF-IDF vector ─────────────────────────────────────────────────────────
  top100   <- vocab[1:min(100, length(vocab))]
  word_tf  <- table(clean) / max(length(clean), 1)
  tfidf_vec <- setNames(rep(0, length(top100)), top100)

  for (w in names(word_tf)) {
    if (w %in% top100) {
      idf_val <- idf_ref %>% filter(word == w) %>% pull(idf)
      idf_val <- ifelse(length(idf_val) == 0, 0, idf_val[1])
      tfidf_vec[w] <- as.numeric(word_tf[w]) * idf_val
    }
  }

  # ── Dense features ────────────────────────────────────────────────────────
  review_length   <- str_count(review_text, "\\S+")
  exclaim_count   <- str_count(review_text, "!")
  caps_ratio      <- str_count(review_text, "[A-Z]") / max(nchar(review_text), 1)
  avg_word_len    <- ifelse(length(clean) > 0,
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

  # ── Assemble & predict ────────────────────────────────────────────────────
  all_features <- c(tfidf_vec, dense_vec)
  feature_row  <- setNames(rep(0, length(model_colnames)), model_colnames)
  matched      <- intersect(names(all_features), model_colnames)
  feature_row[matched] <- all_features[matched]
  feature_df <- as.data.frame(t(feature_row))

  prediction  <- predict(rf_model, feature_df)
  probability <- predict(rf_model, feature_df, type = "prob")

  list(
    verdict     = as.character(prediction),
    fake_prob   = round(probability[1, "Fake"]    * 100, 1),
    genuine_prob = round(probability[1, "Genuine"] * 100, 1),
    review_length = review_length,
    exclaim_count = exclaim_count,
    caps_ratio    = round(caps_ratio * 100, 2)
  )
}

#* Health check
#* @get /health
function() {
  list(status = "ok", model_features = length(model_colnames))
}
