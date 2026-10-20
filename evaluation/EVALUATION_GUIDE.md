# Evaluation guide

Run from the repository root using Python 3.10. Install the base dependencies, then `python -m pip install -r requirements-evaluation.txt`. Evaluators download datasets and model weights as needed; network access and disk/RAM are required.

```powershell
python evaluation/evaluate_sentiment.py
python evaluation/evaluate_classifier.py
python evaluation/evaluate_classifier.py --sample-size 100
python evaluation/evaluate_classifier.py --sample-size 200 --output-suffix experiment_name
python evaluation/evaluate_summarizer.py
python evaluation/evaluate_ner.py
python evaluation/evaluate_keywords.py
python evaluation/evaluate_common_news.py --prepare --sample-size 150
python evaluation/evaluate_common_news.py --evaluate-summary --sample-size 10
```

Outputs are under `evaluation/results/`. Do not treat existing result JSON as a fresh rerun. Sentiment metrics exclude inference failures but include requested and scored counts. Classifier uses four AG News labels and accepts `--sample-size`; the default 500 can be slow on CPU. Summarization uses the CNN/DailyMail test split and ROUGE. NER uses CoNLL-2003 and exact entity span/type matching; LOC→GPE is explicitly approximate. Keyword matching is exact normalized phrase match at K=5.

The classifier `--output-suffix` option writes a separate JSON/CSV pair so experiments do not overwrite the baseline. NER, keyword, and classifier evaluators accept `--dataset-arrow PATH` to read a locally cached split Arrow file without acquiring the Hugging Face datasets cache lock. The file must be the intended split and its provenance should be recorded; normal dataset loading remains the default.

The common-news preparation command reads `abisee/cnn_dailymail`, config `3.0.0`, split `test`, and deterministically sorts records by SHA-256 of record `id`, selecting N. It writes article text/reference summaries and a duplicate CSV template for annotations. Blank labels must remain blank until supplied by a human annotator. See `COMMON_NEWS_EVALUATION.md` for policy. Without annotations, common task metrics are reported as pending, not zero or successful.

If imports fail, check the evaluation extra packages. If dataset/model loading fails, record the exact environment/network error; do not replace the model or dataset silently.
