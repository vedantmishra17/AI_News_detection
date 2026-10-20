# Project audit

Audit basis: source and tracked result files on `main` at the start of this work. The working tree was clean; local `.env` was not opened or changed.

## Architecture and existing functionality

`app.py` calls the NewsAPI and Google News RSS collectors, deduplicates article records, runs cached sentiment/topic/summarization pipelines and spaCy NER, and renders five tabs with Streamlit and Plotly. `keywords.py` computes frequency and corpus TF-IDF terms. `utils/cache.py` provides a JSON file cache, although the dashboard uses Streamlit `cache_data` for collection instead. Existing evaluation scripts covered SST-2, AG News, CNN/DailyMail, and CoNLL-2003; result JSON/CSV files were present.

## Confirmed issues and corrections made

- NewsAPI construction raised on missing credentials, preventing RSS-only operation. It now disables itself cleanly; fetch methods clamp page sizes and report failures without exposing key values.
- RSS retrieval had no explicit timeout and claimed snippets as full content. It now uses bounded HTTP requests and leaves full content empty while retaining the feed description.
- Collector outputs had loose date/field handling. A shared normalizer now supplies stable fields and nullable parsed dates.
- The production sentiment fallback imported TextBlob, but it was missing from `requirements.txt`; TextBlob is now pinned and installed in the project environment.
- Sentiment evaluation loaded the transformer but called production inference, which could silently fall back to TextBlob. It now invokes the loaded pipeline directly and records failures.
- Article title, URL, source, description, summary, and trending terms were inserted into raw HTML. Dynamic fields are HTML-escaped and only HTTP(S) links are rendered as links.
- Summarizer truncated characters rather than tokens. It now truncates with the tokenizer and adjusts generation bounds to input length.
- README omitted evaluation limitations and actual runtime details; it is aligned to current code.
- Per-source records could exceed the dashboard's requested maximum after merging RSS and NewsAPI results; the merged results now honor the slider limit.
- Classifier evaluation ran sequentially and had no short-run option. It now batches with a configurable sample size and records failed inferences.
- NER evaluation did not save predictions or clearly define its entity metric. It now computes exact span/type matches, saves CSV predictions and per-entity counts, maps LOC to GPE as an approximation, and excludes MISC.
- Inspec configuration/fields had not been inspected. The evaluator uses its documented `generation` config, `document`, and `extractive_keyphrases` fields.

## Current remaining risks / needs validation

- Keyword extraction is word-frequency ranking despite the module comment mentioning RAKE; TF-IDF corpus utility is separate. This may confuse maintainers but is not a runtime failure.
- The Streamlit collection cache raises when selected sources all fail, preventing failed empty responses from being cached. Partial results can still be cached when at least one usable source returns records.
- Inference failures can be swallowed by production NLP fallbacks; the dashboard currently lacks per-article backend provenance.
- Large uncached transformer models can exhaust hosting memory or delay first render.

## Master-prompt requirement status

| Area | Status | Evidence or remaining work |
|---|---|---|
| Existing application, schema, collectors, preprocessing, deduplication, security fixes | Completed for the reviewed code paths | Offline fixtures and empty-state Streamlit AppTest pass; no live feed was used. Collection-quality timing/count telemetry is not comprehensive. |
| Individual sentiment, topic, summary, NER, and keyword benchmarks | Partially completed | Actual dataset runs and prediction files exist. SST-2 and CoNLL used 500 records; AG News used 100 and CNN/DailyMail used 10 because CPU throughput made larger fresh runs impractical. A separate 200-example classifier attempt was stopped after 8 inferences in about a minute; candidate-label description experiments were not practical at the observed throughput. |
| Common-news preparation and summary evaluation | Partially completed | 150 deterministic CNN/DailyMail records and an annotation worksheet exist; 10 summaries were scored. The dataset has no gold labels for the other tasks. The evaluator now scores valid human labels if supplied, but none are present. |
| Dashboard validation | Partially completed | Local server health and HTTP 200 were verified. Offline AppTest covers empty and populated tabs, mocked Fetch/Analyze, duplicate removal, partial/all-source failures, and NLP failure. Live source requests failed with network `ConnectionError`; model/resource behavior during a full live fetch remains unverified. |
| Security and dependency review | Completed for local checks | No tracked `.env`; placeholders only; escaping, URL scheme checks, offline import and dependency checks pass. Routine vulnerability review remains before public deployment. |
| Deployment preparation | Prepared, not verified | Setup and secret instructions exist. No cloud build, deployment, or deployed smoke test was performed. |
| Git safety | Completed | Existing changes are preserved; no files are deleted; no commit or push was made. |

## Dependencies and evaluation

Local and project `.venv` Python are 3.10.10. `rouge-score` is declared in `requirements-evaluation.txt`; the NER evaluator uses its own exact span/type calculation and does not require `seqeval`. Fresh results are reported in `docs/EVALUATION_REPORT.md`; historical metrics and predictions are retained with `_historical` filenames. Sentiment and NER historical values were reproduced. AG News and summarization fresh runs used smaller samples due CPU cost. Four-class AG News does not validate all ten app categories. CNN/DailyMail supplies summary references only; task labels for sentiment/topic/NER/keywords remain unannotated. The common-news evaluator now reads the annotation worksheet and can score supplied labels; no such labels are currently present.

## Security, performance, deployment

`.env` is ignored and is not tracked. A placeholder `.env.example` was added. No credentials were printed. Raw HTML rendering existed in the dashboard; dynamic collector values are now escaped. RSS URLs have protocol validation and timeouts. The installed NewsAPI client's request timeout is 30 seconds. Disk cache helper is not called by app code; Streamlit cache is used for fetches, and `lru_cache` is used for models. A Streamlit `AppTest` verified empty-state startup; a plain Python import is not a valid Streamlit runtime check because it has no session context. No deployment was performed. Community Cloud compatibility/resource fit remains to be validated with a real build.

## Remaining work

Human annotation of the common-news records, full-size CPU-expensive benchmarks, live collector checks, and deployment build/resource validation remain outstanding. See `docs/VALIDATION_REPORT.md` for executed commands and actual outcomes.
