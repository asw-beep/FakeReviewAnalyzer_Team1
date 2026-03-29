# =============================================================================
# STAGE 2: Text Preprocessing
# Review Trust Analyzer | CSS-321 | Group 1
# =============================================================================
# OBJECTIVE: Convert raw review text into clean, normalized tokens
# ready for TF-IDF and LDA. Build a reusable clean() function
# applied to both datasets.
# =============================================================================

library(tidyverse)
library(tidytext)
library(SnowballC)

# --------------------------------------------------------------------------
# 2.1  LOAD STAGE 1 OUTPUT
# --------------------------------------------------------------------------
cat("Loading Stage 1 data...\n")
amazon <- readRDS("data/amazon_stage1.rds")
fake   <- readRDS("data/fake_stage1.rds")
cat("Amazon:", nrow(amazon), "rows\n")
cat("Fake:  ", nrow(fake),   "rows\n")

# --------------------------------------------------------------------------
# 2.2  DEFINE CLEAN FUNCTION
# --------------------------------------------------------------------------
# Steps: lowercase → strip URLs/numbers/punctuation → tokenize →
#        remove stopwords → stem → rejoin into single string

stopwords_list <- stop_words$word  # tidytext built-in English stopwords

clean_text <- function(text) {
  text %>%
    str_to_lower() %>%                            # lowercase
    str_remove_all("https?://\\S+") %>%           # remove URLs
    str_remove_all("[^a-z\\s]") %>%               # keep only letters + spaces
    str_squish() %>%                              # remove extra whitespace
    str_split("\\s+") %>%                         # tokenize
    lapply(function(tokens) {
      tokens <- tokens[!(tokens %in% stopwords_list)]  # remove stopwords
      tokens <- tokens[nchar(tokens) > 2]              # remove very short words
      tokens <- wordStem(tokens, language = "english") # stem
      paste(tokens, collapse = " ")                    # rejoin
    }) %>%
    unlist()
}

# --------------------------------------------------------------------------
# 2.3  APPLY TO BOTH DATASETS
# --------------------------------------------------------------------------

# --- Amazon ---
cat("\nCleaning Amazon reviews (this takes ~2-3 mins for 80K rows)...\n")
start <- Sys.time()
amazon <- amazon %>%
  mutate(clean_text = clean_text(reviewText))
cat("Done in", round(difftime(Sys.time(), start, units = "mins"), 1), "mins\n")

# --- Fake ---
cat("Cleaning Fake reviews...\n")
fake <- fake %>%
  mutate(clean_text = clean_text(reviewText))
cat("Done.\n")

# --------------------------------------------------------------------------
# 2.4  SPOT CHECK — print 5 random before/after pairs
# --------------------------------------------------------------------------
cat("\n=== Spot Check: Before vs After Cleaning (Amazon) ===\n")
set.seed(42)
check <- amazon %>%
  sample_n(5) %>%
  select(reviewText, clean_text)

for (i in 1:5) {
  cat("\n--- Review", i, "---\n")
  cat("BEFORE:", str_trunc(check$reviewText[i], 120), "\n")
  cat("AFTER: ", str_trunc(check$clean_text[i],  120), "\n")
}

# --------------------------------------------------------------------------
# 2.5  REMOVE EMPTY CLEAN TEXT ROWS
# --------------------------------------------------------------------------
amazon <- amazon %>% filter(nchar(clean_text) > 5)
fake   <- fake   %>% filter(nchar(clean_text) > 5)

cat("\nAmazon after removing empty clean rows:", nrow(amazon), "\n")
cat("Fake after removing empty clean rows:  ", nrow(fake),   "\n")

# --------------------------------------------------------------------------
# 2.6  SAVE FOR NEXT STAGE
# --------------------------------------------------------------------------
saveRDS(amazon, "data/amazon_stage2.rds")
saveRDS(fake,   "data/fake_stage2.rds")

cat("\n=== Stage 2 Complete ===\n")
cat("Saved: data/amazon_stage2.rds\n")
cat("Saved: data/fake_stage2.rds\n")
cat("Next: run 03_features.R\n")