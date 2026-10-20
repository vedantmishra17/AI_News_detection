# Security review

## Findings and changes

- The local `.env` is ignored by Git and was not read. `git ls-files .env` returned no tracked path. `.env.example` contains only a placeholder.
- Dashboard article-card content was interpolated into `unsafe_allow_html=True` markup. Title, URL, source, description, summary, and trending terms are now escaped. Article links accept only `http://` and `https://`; other values render as `#`.
- RSS requests now require HTTP(S), use a 5-second connect and 15-second read timeout, and cap item count. NewsAPI errors log exception types and API response codes only; the API key is not included.
- Feed and NewsAPI fields are treated as untrusted text and normalized before downstream processing.

## Remaining risks

- NewsAPI calls use the installed client's 30-second request timeout; the collector does not currently expose a per-call timeout setting.
- Model downloads and collector endpoints require outbound network access. Arbitrary custom RSS feeds can cause server-side requests if that API is exposed to untrusted users; currently it is a code-level method, not a user input in the dashboard.
- Streamlit charts and text should continue to use standard rendering APIs. Review future HTML changes for escaping.
- `.env` is absent from the tracked tree and reachable Git history. A credential-shaped string scan of reachable revisions returned no matching filenames. No history was rewritten. This scan cannot prove secrets were never shared outside this repository; if a real key was ever exposed elsewhere, revoke it in NewsAPI.
- Pinned dependencies are older versions and require routine vulnerability review before public deployment.
