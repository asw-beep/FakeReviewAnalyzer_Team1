# =============================================================================
# STAGE 9: Word Clouds — Fake vs Genuine (Distinctive Vocabulary)
# Review Trust Analyzer | CSS-321 | Group 1
# =============================================================================

library(tidyverse)
library(tidytext)
library(wordcloud)
library(RColorBrewer)

cat("Loading data...\n")
fake <- readRDS("dataset/fake_stage4.rds")

dir.create("figures", showWarnings = FALSE)

# --------------------------------------------------------------------------
# Find DISTINCTIVE words per class using log odds ratio
# Words that appear proportionally MORE in fake vs genuine and vice versa
# --------------------------------------------------------------------------
word_counts <- fake %>%
  select(label_name, clean_text) %>%
  unnest_tokens(word, clean_text) %>%
  count(label_name, word) %>%
  filter(nchar(word) > 2)

# Total words per class
total_words <- word_counts %>%
  group_by(label_name) %>%
  summarise(total = sum(n))

# Log odds ratio — how much more likely is each word in fake vs genuine
log_odds <- word_counts %>%
  left_join(total_words, by = "label_name") %>%
  pivot_wider(names_from = label_name,
              values_from = c(n, total),
              values_fill = 0) %>%
  mutate(
    # Add 1 to avoid log(0)
    fake_ratio    = (n_Fake + 1)    / (total_Fake + 1),
    genuine_ratio = (n_Genuine + 1) / (total_Genuine + 1),
    log_odds      = log(fake_ratio / genuine_ratio)
  )

# Distinctive fake words — high positive log odds
distinctive_fake <- log_odds %>%
  filter(n_Fake > 10) %>%
  arrange(desc(log_odds)) %>%
  slice_head(n = 150)

# Distinctive genuine words — high negative log odds
distinctive_genuine <- log_odds %>%
  filter(n_Genuine > 10) %>%
  arrange(log_odds) %>%
  slice_head(n = 150)

# --------------------------------------------------------------------------
# Word Cloud — Distinctive FAKE words
# --------------------------------------------------------------------------
cat("Generating fake reviews word cloud...\n")
png("figures/fig20_wordcloud_fake.png", width = 900, height = 650, res = 130)
par(bg = "#1a1a2e", mar = c(0, 0, 3, 0))
wordcloud(
  words        = distinctive_fake$word,
  freq         = distinctive_fake$n_Fake,
  max.words    = 100,
  scale        = c(4, 0.5),
  colors       = brewer.pal(9, "Reds")[4:9],
  random.order = FALSE,
  rot.per      = 0.15
)
title("Fake Reviews — Distinctive Vocabulary",
      col.main = "white", cex.main = 1.3)
dev.off()
cat("Saved: figures/fig20_wordcloud_fake.png\n")

# --------------------------------------------------------------------------
# Word Cloud — Distinctive GENUINE words
# --------------------------------------------------------------------------
cat("Generating genuine reviews word cloud...\n")
png("figures/fig21_wordcloud_genuine.png", width = 900, height = 650, res = 130)
par(bg = "#0d1f0d", mar = c(0, 0, 3, 0))
wordcloud(
  words        = distinctive_genuine$word,
  freq         = distinctive_genuine$n_Genuine,
  max.words    = 100,
  scale        = c(4, 0.5),
  colors       = brewer.pal(9, "Greens")[4:9],
  random.order = FALSE,
  rot.per      = 0.15
)
title("Genuine Reviews — Distinctive Vocabulary",
      col.main = "white", cex.main = 1.3)
dev.off()
cat("Saved: figures/fig21_wordcloud_genuine.png\n")

# --------------------------------------------------------------------------
# Comparison Table — now shows truly distinct words
# --------------------------------------------------------------------------
cat("\n=== Top 15 DISTINCTIVE Words: Fake vs Genuine ===\n")
comparison <- bind_cols(
  distinctive_fake    %>% slice_head(n = 15) %>% select(Fake_Word = word),
  distinctive_genuine %>% slice_head(n = 15) %>% select(Genuine_Word = word)
)
print(comparison)

cat("\n=== Stage 9 Complete ===\n")
cat("Word clouds now show DISTINCTIVE vocabulary per class\n")
cat("(words disproportionately common in fake vs genuine)\n")