# Dataset Description

## Dataset 1: Amazon Fine Food Reviews

- **Source:** [https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Description:** Contains reviews of fine foods from Amazon spanning over 10 years. Includes product and user information, ratings, and review text. Used as the "genuine" review source in our pipeline.
- **Number of instances:** 568,454 (80,000 sampled for processing)
- **Number of attributes:** 10
- **Key attributes:**
  | Attribute | Description |
  |-----------|-------------|
  | `Id` | Unique row identifier |
  | `ProductId` | Amazon product ASIN |
  | `UserId` | Reviewer identifier |
  | `ProfileName` | Reviewer display name |
  | `HelpfulnessNumerator` | Number of helpful votes |
  | `HelpfulnessDenominator` | Total votes on helpfulness |
  | `Score` | Star rating (1–5) |
  | `Time` | Unix timestamp of review |
  | `Summary` | Short review summary |
  | `Text` | Full review text |

### How to Download

1. Go to the Kaggle link above
2. Download `Reviews.csv`
3. Rename it to `amazon_reviews.csv`
4. Place it in the `data/` folder

---

## Dataset 2: Fake Reviews Dataset

- **Source:** [https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset)
- **Description:** Contains both computer-generated (fake) and original (genuine) product reviews. Each review is labeled, making it ideal for supervised classification.
- **Number of instances:** ~40,000
- **Number of attributes:** 4
- **Key attributes:**
  | Attribute | Description |
  |-----------|-------------|
  | `category` | Product category |
  | `rating` | Star rating (1–5) |
  | `text_` | Full review text |
  | `label` | `CG` (computer-generated/fake) or `OR` (original/genuine) |

### How to Download

1. Go to the Kaggle link above
2. Download `fake reviews dataset.csv`
3. Rename it to `fake_reviews.csv`
4. Place it in the `data/` folder
