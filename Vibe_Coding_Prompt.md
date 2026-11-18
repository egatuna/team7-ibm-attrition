# Vibe-coding prompt — Team 7, Challenge 4 (final version used to generate streamlit_app.py)
Deployment target: Streamlit Community Cloud, per the instructor's deployment update (Option 1), because Plotly Cloud's free tier no longer allows public links or external APIs.

**Role & goal**
You are a senior Python data-app developer. Build a single-file **Streamlit** application with Plotly charts, `streamlit_app.py`, that will be **deployed on Streamlit Community Cloud from a public GitHub repository** (map `st.secrets` to environment variables; cache the K-Means fits with `st.cache_resource`; no local paths other than `IBM_Dataset.csv` in the same folder). The user is Dr. Sarah Chen, CHRO of IBM, and her HR leadership team. The app must help them see **what kinds of employees are at risk of leaving** by segmenting employees with **K-Means** and comparing each segment's **observed attrition rate**. It must not predict individual attrition.

**Data** — `IBM_Dataset.csv`, 1,470 rows: Age, Attrition (Yes/No), Department (HR / R&D / Sales), Education (1–5 ordinal), Gender (Female/Male), JobLevel (1–5 ordinal), JobSatisfaction (1–4 ordinal), MonthlyIncome, TotalWorkingYears, WorkLifeBalance (1–4 ordinal), YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager. If an `OverTime` (Yes/No) column exists, include it automatically.

**Preprocessing (explain each choice in comments)**
1. Use every variable except Attrition as a clustering input. **Attrition is never a clustering input**; use it only to compute and compare each cluster's attrition rate.
2. Keep continuous variables numeric. Keep ordinal variables as ordered integers. Encode Gender (and OverTime) as 0/1. One-hot encode Department (it has no order).
3. Standardize all encoded features with `StandardScaler`, because K-Means uses Euclidean distance and income in dollars would otherwise dominate.

**Modeling**
- Fit K-Means for k = 2…10 (`n_init=20`, `random_state=42`) at start-up. Store inertia and silhouette for each k.
- Renumber clusters so **Cluster 1 = highest observed attrition**.
- Default k = 4, with a slider so users can explore any k from 2 to 10.
- Give each cluster a business label generated from its dominant department and seniority, e.g. "Sales Front Line", "Senior Loyalty Anchors".

**Layout** (navy #0F2537 header, Purdue gold #CFB991 accents, white cards on light grey)
- Header plus a **responsible-use banner**: segments are for retention planning only, never hiring, promotion or discipline; Attrition mixes voluntary and involuntary exits.
- k slider and a KPI row: employees, overall attrition, silhouette for the selected k, highest-risk segment and lowest-risk segment.
- Tabs:
  1. **Overview** — attrition-rate bar by segment with the IBM average line; segment-size donut; 3D PCA scatter.
  2. **Model Quality** — side-by-side Elbow and Silhouette charts for k = 2–10 with the selected k highlighted, a metrics table and a written justification of the chosen k.
  3. **Cluster Profiles** — profile table (size, attrition, means, % male, department mix); centroid z-score heatmap; drill-down showing the attrition rate within each segment for any variable; plain-language summaries.
  4. **Cluster Simulator** — one input per clustering variable, **pre-filled with the median or most frequent value**. Output: the assigned cluster (nearest centroid), its observed attrition rate and profile, a distance-to-each-centroid bar chart, assignment confidence (margin over the runner-up) and the case plotted in 3D. Include a "Junior Sales example" button (JobLevel 2, low satisfaction, bad WLB, 2 years at IBM) and a "Reset" button.
  5. **Interpretability** — the K-Means steps and a preprocessing table showing how the clustering is implemented.
  6. **Explainability** — for the simulated case or any employee ID: distances to every centroid, plus a driver chart giving each feature's squared-distance advantage versus the runner-up cluster.
  7. **AI Assistant** — a chatbot grounded in the app's results. Call Google Gemini (free tier; `GEMINI_API_KEY`, model from `GEMINI_MODEL` if set, otherwise auto-discover the available Flash models (lite models first), temperature 0.2) through the REST API, with a JSON context of cluster profiles, Elbow/Silhouette metrics, attrition by factor and the current simulation. OpenAI (`OPENAI_API_KEY`) is an optional alternative. The system prompt says: answer only from the context, quote numbers, never predict an individual's attrition, and restate the responsible-use limits. **If there is no key or the call fails, fall back to a rule-based engine** that answers from the same data and show why the AI model was unavailable, so the live demo never breaks. Include suggested-question buttons and show which engine answered.

**Quality bar**: no errors in the browser console; every chart updates when k changes; numbers in the chatbot match the profile table; a `requirements.txt` with streamlit, plotly, pandas, numpy, scikit-learn, requests and openai, all with pinned versions.

---
### Tests and improvements made during development (actual)
1. **Data check:** the provided IBM_Dataset (.xlsx and .csv) has **no OverTime column** (12 inputs, not 14). The prompt was changed to include OverTime automatically if it appears, and the app banner and chatbot now state the gap.
2. **k sweep:** we compared plain standardization with log-transformed tenure and income. Plain z-scores gave higher silhouettes for every k from 2 to 6 (k=4: 0.220 vs 0.197), so we kept them.
3. **Chatbot test:** the question "Why did you choose k=4?" fell through to the help message. We broadened the k/silhouette intent matching and re-tested all suggested questions.
4. **UI test (automated screenshots of all 7 tabs):** distance labels were clipped at the chart edge, so we added axis padding. No console errors.
5. **Explainability readability (revision 2):** we renamed encoded features (e.g. `Dept_Sales` became "Works in Sales"), added a plain-English summary, and added a table comparing the employee with the assigned and runner-up segment averages, so a non-technical HR user can follow it. Redeployed to Plotly Cloud.
6. **Chatbot provider (revision 3):** live testing showed OpenAI returning 429 (no credits on the account), and the error was now visible in the UI. We switched to Google Gemini's free tier through REST, added a 12–15 s timeout so a stuck call cannot freeze the app, and kept OpenAI as an optional alternative.
7. **Gemini model discovery (revision 3 on the new app):** the first live test hit 503 (model overloaded), and then 404s as model names changed. The app now asks Google which Flash models the key can use, tries the fast "lite" models first, and moves on if one is busy. Verified live: "highest attrition" and "explain my simulation" answers matched the app (C1 22.5%, 87 of 387, margin 0.95).
8. **Deployment platform (final):** the instructor confirmed that Plotly Cloud's free tier now blocks public links and external APIs. We chose Option 1 and ported the app to Streamlit: the same analysis engine, charts and chatbot, with Streamlit tabs, widgets with session state for the simulator, and `st.chat_input` for the assistant. Deployed on Streamlit Community Cloud with `GEMINI_API_KEY` in Secrets. The original Dash version is kept in `plotly_version/` for reference.
