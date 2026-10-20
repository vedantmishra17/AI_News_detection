# Evaluation report

Historical result files were copied to `*_historical.*` before fresh runs. Results below are actual runs from this work; the 500-example classifier history was not completed because of CPU cost.

| Task | Dataset / model | Current result | Scope limit |
|---|---|---:|---|
| Sentiment | GLUE SST-2 validation, first 500; DistilBERT SST-2 | Accuracy 91.20%; positive P 90.18%, R 93.58%, F1 91.85% | Exact historical match with direct-transformer inference. Movie reviews do not establish news sentiment quality. |
| Topic | AG News test, first 100; BART-MNLI, four categories | Accuracy 82.00%; macro F1 78.12%; weighted F1 81.21%; Business recall 41.67% | Full 500 run stopped for CPU cost. First-100 subset is not directly comparable with historical first-500. Candidate labels unchanged. |
| Summary | CNN/DailyMail 3.0.0 test, first 10; BART-large-CNN | ROUGE-1 41.72%; ROUGE-2 20.95%; ROUGE-L 32.17% | Full 100 run stopped for CPU cost. Corrected tokenizer-aware protocol differs from historical 1024-character truncation. ROUGE is not factuality. |
| NER | CoNLL-2003 test, first 500; spaCy `en_core_web_sm` | Exact-span micro P 67.95%, R 54.03%, F1 60.19%; classwise macro F1 55.49%, support-weighted F1 60.48% | Reproduced historical values; class averages are computed over PERSON/ORG/GPE exact spans. LOC→GPE is an approximation. MISC is excluded. |
| Keywords | Inspec `generation`, test 500; frequency baseline | P@5 3.48%; R@5 3.59%; F1@5 3.19%; MRR 6.97% | Exact phrase match on academic abstracts; does not establish performance on news. |

The sentiment evaluator calls DistilBERT directly and records inference failures. Summarization evaluation now tokenizes and truncates to 1024 tokens. Keyword evaluation maps Inspec `document` (a token list, joined with spaces) to input and `extractive_keyphrases` (a list of phrases) to gold; normalization is Unicode casefold, alphanumeric filtering, whitespace collapse, then exact phrase match. In the 500-record sample, 84.56% of the 3,283 gold phrases are multiword (mean 2.23 words), while the production frequency baseline emits only single-token terms. This granularity mismatch depresses exact phrase scores; the baseline also ranks by frequency and does not extract phrases. The score is retained as the defined exact-phrase metric; no relaxed overlap score is substituted or presented as a model-quality improvement.

AG News business/finance remains the weakest topic class: in the fresh first-100 sample recall was 5/12 (41.67%), with six of twelve examples mapped to World News. The historical first-500 artifact reports 25.47% recall. A new first-200 run processed only 8 examples in about one minute after model loading and was stopped; no metrics were written. Candidate wording comparisons were not run because their inference cost is similar. No improvement is claimed. NER organization F1 is 26.48% (47 true positives, 120 false positives, 141 false negatives) on this CoNLL slice; no replacement model was tested.

## Common news subset

The workflow prepared 150 reproducibly selected records from CNN/DailyMail 3.0.0 test (SHA-256 ordering by ID). Available dataset fields are `id`, `article`, and `highlights`; no sentiment/topic/entity/keyword labels are provided. Summary ROUGE on the first 10 prepared records: ROUGE-1 38.72%, ROUGE-2 18.08%, ROUGE-L 27.64%. Sentiment/topic/NER/keyword task counts are zero because no human annotations have been supplied. These tasks remain pending. No model-generated labels are used as gold.
