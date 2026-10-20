# Validation report

Environment: Windows, Python 3.10.10 (system and `.venv`). Credential values were not printed. A configured NewsAPI key was used internally for a live request; outbound network access failed.

## Executed checks

| Command | Result |
|---|---|
| `python -m compileall -q app.py src evaluation` | Pass. |
| `python -m pip check` | Pass; no broken requirements. |
| `python -m pip install textblob==0.20.1` | Pass in the project `.venv`; the missing production fallback dependency was installed and exercised by the mocked-load smoke check. |
| `python evaluation/smoke_checks.py` | Pass: Unicode/control cleaning, normalized schema, URL/title dedupe, malformed NewsAPI fixture, RSS HTML stripping, missing API key behavior, empty input, mocked NLP model-load fallbacks, keyword extraction, entity aggregation, mocked common-news annotation scoring. Streamlit `AppTest` empty-state render passed. |
| `python evaluation/dashboard_smoke_checks.py` | Pass: populated all five tabs; mocked Fetch/Analyze; duplicate URL removal; missing/invalid fields and unsafe URL; all-source failure; partial source failure; and NLP failure. AppTest teardown emitted `expire_cache` coroutine runtime warnings. |
| Local `streamlit run app.py --server.headless true --server.port 8508`; `GET /_stcore/health`, `GET /` | Pass; health returned `ok`, page returned HTTP 200. Server was stopped after verification. |
| Live NewsAPI headlines, `page_size=1` | Attempted with configured credentials; returned zero articles and `ConnectionError` because outbound network access is blocked. Key value was not printed. |
| Live Google News RSS Technology feed, `max_items=2` | Attempted; returned zero articles and `ConnectionError` because outbound network access is blocked. |
| Import all six evaluator modules | Pass. |
| `python evaluation/evaluate_sentiment.py` | Pass, 500 scored, 0 failures; accuracy .9120 and positive F1 .9185. |
| `python evaluation/evaluate_classifier.py --sample-size 100` | Pass, 100 scored, 0 failures; accuracy .8200, macro F1 .7812, weighted F1 .8121. A 500-example run was stopped after observing low CPU throughput; no 500-example fresh score is claimed. |
| `python evaluation/evaluate_summarizer.py --sample-size 10` | Pass, 10 scored, 0 failures; ROUGE-1 .4174, ROUGE-2 .2095, ROUGE-L .3217. A 100-example run was stopped due CPU cost. |
| NER evaluator with cached `conll2003-test.arrow`, sample size 500 | Pass; exact span/type micro F1 .6019, macro F1 .5549, weighted F1 .6048. `Dataset.from_file` read the existing Arrow file; no cache lock was removed or changed. |
| Keyword evaluator with cached `inspec-test.arrow`, sample size 500 | Pass; P@5 .0348, R@5 .0359, F1@5 .0319, MRR .0697. The source schema has token-list `document` and phrase-list `extractive_keyphrases`; 84.56% of gold phrases in this sample are multiword. |
| Classifier evaluator, sample size 200, separate output suffix | Stopped after 8/200 inferences in about one minute due CPU throughput. No metrics file was written; the existing 100-example baseline remains unchanged. |
| `python evaluation/evaluate_common_news.py --prepare --sample-size 150` | Pass. Wrote deterministic 150 article subset, annotation worksheet, and manifest with selected IDs. No non-summary labels created. |
| `python evaluation/evaluate_common_news.py --evaluate-summary --sample-size 10` | Pass, 10 articles; ROUGE-1 .3872, ROUGE-2 .1808, ROUGE-L .2764. Other common-news tasks remain annotation pending. |
| `git ls-files .env` | No tracked `.env` path. |
| `git log --all -- .env`; credential-shaped `git grep -l` over all reachable revisions | No `.env` history path or matching filename was found; no secret values were printed. |

## Skipped / not verified

- Live NewsAPI/RSS requests: attempted; both failed with `ConnectionError` due outbound network restrictions. Offline mocked collector paths passed.
- Dashboard interaction with live collection and production NLP models: not run because both live collectors failed. Local server startup and mocked populated/failure flows passed.
- Full 500 AG News and 100 CNN/DailyMail runs: stopped for CPU throughput; see fresh smaller-sample results and historical artifacts separately.
- Human annotation for sentiment/topic/entities/keywords on the common-news subset: not supplied, so those metrics are not computed.
- Clean-environment dependency installation and deployment build/runtime: not run; outbound network access is blocked and the pinned Torch/Transformers stack is large. `pip check` passed in the existing `.venv`; no deployment is claimed.
- Cold/warm timing and memory benchmark suite: not measured; classifier process resident memory observed around 1.6–1.8 GB, summarizer around 2.3–2.6 GB during CPU evaluation. These are point observations, not deployment capacity estimates.

The NER cache-lock failure came from trying to acquire a lock outside the writable workspace. Direct read-only loading of the cached Arrow split avoided the lock, and the 500-record rerun completed. The keyword rerun used the same direct-Arrow approach. The larger classifier attempt confirmed CPU throughput was too low for a useful expanded run here.
