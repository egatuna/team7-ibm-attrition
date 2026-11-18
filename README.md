# IBM Workforce Segmentation Studio (QM 389 Challenge 4, Team 7)

This Streamlit app segments 1,470 IBM employees with K-Means (k = 2–10, Elbow and Silhouette) and compares observed attrition across the segments. It includes a cluster simulator, interpretability and explainability tabs, and an AI assistant grounded in the cluster results (Google Gemini free tier, with a built-in fallback if Gemini is unavailable).

Team: Eann Gatuna, Arianna Dilling, Alexis Coberg, Avery Lowry.

**Run locally:** `pip install -r requirements.txt`, then `streamlit run streamlit_app.py`.
**Chatbot:** add `GEMINI_API_KEY` under the app's Settings → Secrets.

Responsible use: the segments are for retention planning only. Attrition combines voluntary and involuntary departures.
