# Deployment preparation

No deployment has been performed. Streamlit Community Cloud is a plausible target for this Streamlit app, but the pinned Torch/Transformers stack and three large model pipelines have not been built or measured on that host. Cloud supports dependency files and secrets through its deployment settings; choose a Python version compatible with the pinned packages and verify the build log.

1. Connect the existing GitHub repository `vedantmishra17/AI_News_detection` in Streamlit Community Cloud.
2. Select the intended branch (`main`) and entrypoint `app.py`.
3. In app settings, select Python 3.10 if still supported by the platform and compatible with these pins; otherwise update and validate dependencies together.
4. Add `NEWS_API_KEY` in the platform's Secrets panel. Do not commit `.env` or `secrets.toml`.
5. Confirm the Linux build can install PyTorch and spaCy. If `en_core_web_sm` is not available, add a reproducible installation step supported by the selected platform.
6. Open the deployed app and check startup, RSS-only fetch, configured NewsAPI fetch, model cold start, every dashboard tab, empty results, and recovery after an external feed failure.

The app downloads model weights at runtime on first use. The local Hugging Face cache contains about 1.63 GB of files for BART-MNLI and 1.63 GB for BART-CNN; these are disk-cache sizes, not a clean deployment download measurement. Previous CPU evaluation observed roughly 1.6–1.8 GB process memory for classifier inference and 2.3–2.6 GB for summarization. The current environment blocked outbound requests and did not allow a clean install or cloud build; model cold-start and cloud CPU/memory behavior remain unverified. Do not claim deployment readiness until measured. If the build or runtime exceeds limits, record measurements before considering a model change.

Current Streamlit instructions: [dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies), [deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), and [platform limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/status).
