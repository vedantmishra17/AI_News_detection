# Common-news evaluation workflow

CNN/DailyMail 3.0.0 test is the selected common source because its dataset card exposes article IDs, article text, and reference highlights. Hugging Face metadata lists Apache-2.0, while the card's licensing paragraph specifically describes version 1.0.0 and the 3.0 version is non-anonymized. This version-specific licensing ambiguity is recorded in the manifest; verify rights and attribution before redistributing source text.

Run `python evaluation/evaluate_common_news.py --prepare --sample-size 150`. It reads the test split and takes records ordered by SHA-256 of their `id`, making selection reproducible. The resulting `common_news_subset.csv` has the source data and `common_news_annotations.csv` is the annotation worksheet. Both are generated locally and not prefilled with invented labels. `id`, `article`, and `highlights` map to `article_id`, `article_text`, and `reference_summary`; dataset title is absent and left blank.

## Annotation guidance

- Sentiment: use Positive, Negative, or Neutral for the article's overall framing, not the polarity of one quoted source. Use blank if unclear.
- Topic: assign one primary category from the app's ten categories (Politics, Technology, Business & Finance, Health & Medicine, Science, Sports, Entertainment, World News, Environment, Crime & Law); use blank for ambiguous cases and record the reason outside the score file.
- Entities: store JSON list of `{text, label}` objects with exact text spans; labels should be PERSON, ORG, or GPE. Dates are excluded from common-news entity scoring unless a later protocol explicitly adds them.
- Keywords: store a JSON list of concise key phrases grounded in the article; avoid generic terms and duplicates. The evaluator must define K and matching normalization before reporting scores.
- Set `annotation_status` to `annotated` only when the row's applicable labels are reviewed. Put annotator identity/role in `annotator`; do not claim multiple annotators unless used.

The dataset has reference summaries, but no sentiment, topic, NER, or keyword gold labels. Metrics use only valid, non-empty human labels. After annotating the worksheet, run `python evaluation/evaluate_common_news.py --evaluate-annotations`; it reads the worksheet, writes task metrics and `common_news_annotation_predictions.csv`, and leaves tasks without labels pending. For summary overlap, run `python evaluation/evaluate_common_news.py --evaluate-summary --sample-size 10`; it reads the worksheet when present and scores the supplied highlights. Sentiment/topic labels must match the choices above; entity and keyword fields must be JSON arrays. Factual consistency needs separate human review and is not scored by ROUGE.
