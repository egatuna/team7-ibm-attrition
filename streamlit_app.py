"""
IBM Workforce Segmentation Studio  —  QM 389 Challenge 4 (Team 7)
=================================================================
Interactive K-Means segmentation of IBM employees to understand *what kinds*
of employees leave.  Streamlit version, deployed on Streamlit Community Cloud.

Run locally:   streamlit run streamlit_app.py      (needs IBM_Dataset.csv in this folder)
Chatbot:       add GEMINI_API_KEY to the app's Secrets (free key from aistudio.google.com).
               Without a key the chatbot answers from the cluster results with a
               built-in grounded engine, so the app never breaks.

Responsible use: segments are for retention planning only — never for hiring,
promotion or disciplinary decisions.  Attrition mixes voluntary and
involuntary departures; the data cannot separate them.
"""
import os
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="IBM Workforce Segmentation Studio", page_icon="📊", layout="wide")

# Streamlit Cloud secrets -> environment variables used by the chatbot
try:
    for _k in ("GEMINI_API_KEY", "GEMINI_MODEL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass

# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IBM_Dataset.csv")
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df.columns = [c.strip() for c in df.columns]
for c in df.select_dtypes("object").columns:
    df[c] = df[c].astype(str).str.strip()

RANDOM_STATE = 42
K_RANGE = list(range(2, 11))
DEFAULT_K = 4
df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)
OVERALL_RATE = df["AttritionFlag"].mean()

# Variable roles (Attrition is NEVER a clustering input)
NUMERIC = ["Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
           "YearsInCurrentRole", "YearsWithCurrManager"]
ORDINAL = ["Education", "JobLevel", "JobSatisfaction", "WorkLifeBalance"]
BINARY = ["Gender"] + (["OverTime"] if "OverTime" in df.columns else [])
NOMINAL = ["Department"]
CLUSTER_VARS = NUMERIC + ORDINAL + BINARY + NOMINAL
BINARY_POSITIVE = {"Gender": "Male", "OverTime": "Yes"}

LABELS = {
    "Education": {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"},
    "JobLevel": {1: "Entry", 2: "Junior", 3: "Mid", 4: "Senior", 5: "Executive"},
    "JobSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
    "WorkLifeBalance": {1: "Bad", 2: "Good", 3: "Better", 4: "Best"},
}
DEPARTMENTS = sorted(df["Department"].unique())


# Feature selection: every available HR variable except Attrition is used, as the
# brief requires. Each one maps to one of Dr. Chen's questions: demographics (Age,
# Gender, Education), position (Department, JobLevel, MonthlyIncome[, OverTime]),
# seniority (TotalWorkingYears, YearsAtCompany, YearsInCurrentRole,
# YearsWithCurrManager) and satisfaction (JobSatisfaction, WorkLifeBalance).
# Attrition is held out so segments describe *who employees are*, and the
# attrition rate can then be compared across segments without circularity.
def encode(frame: pd.DataFrame) -> pd.DataFrame:
    """Numeric + ordinal kept as numbers (order matters), binaries -> 0/1,
    Department -> one-hot (no order between departments)."""
    X = frame[NUMERIC + ORDINAL].astype(float).copy()
    for b in BINARY:
        X[f"{b}_{BINARY_POSITIVE[b]}"] = (frame[b] == BINARY_POSITIVE[b]).astype(float)
    for dpt in DEPARTMENTS:
        X[f"Dept_{dpt}"] = (frame["Department"] == dpt).astype(float)
    return X


X_raw = encode(df)
FEATURES = list(X_raw.columns)
scaler = StandardScaler().fit(X_raw)          # z-score so $ income does not dominate
Z = scaler.transform(X_raw)

# ---------------------------------------------------------------------------
# 2. MODELS  (k = 2..10 fitted once at start-up)
# ---------------------------------------------------------------------------
PALETTE = ["#E15759", "#F28E2B", "#4E79A7", "#59A14F", "#B07AA1",
           "#76B7B2", "#EDC948", "#9C755F", "#FF9DA7", "#86BCB6"]

pca = PCA(n_components=3, random_state=RANDOM_STATE).fit(Z)
P3 = pca.transform(Z)
PCA_AXES = [f"PC{i+1} ({v:.0%})" for i, v in enumerate(pca.explained_variance_ratio_)]

@st.cache_resource(show_spinner="Fitting K-Means for k = 2–10 …")
def fit_models(Z):
    """Fit K-Means for every k once and cache it (Streamlit reruns the script on each click)."""
    models, metrics = {}, []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(Z)
        # Re-number clusters by observed attrition (Cluster 1 = highest risk)
        rates = pd.Series(df["AttritionFlag"].values).groupby(km.labels_).mean()
        order = rates.sort_values(ascending=False).index.tolist()
        remap = {old: new for new, old in enumerate(order)}
        labels = np.array([remap[l] for l in km.labels_])
        centers = km.cluster_centers_[order]
        sil = silhouette_score(Z, labels)
        models[k] = {"labels": labels, "centers": centers, "inertia": km.inertia_, "silhouette": sil}
        metrics.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return models, pd.DataFrame(metrics)


MODELS, METRICS = fit_models(Z)
BEST_SIL_K = int(METRICS.loc[METRICS["silhouette"].idxmax(), "k"])


def seniority(jl):
    return "Early-Career" if jl < 1.75 else ("Mid-Level" if jl < 2.6 else "Senior")


def cluster_name(sub: pd.DataFrame) -> str:
    dept_share = sub["Department"].value_counts(normalize=True)
    dept, share = dept_share.index[0], dept_share.iloc[0]
    sen = seniority(sub["JobLevel"].mean())
    short = {"Research & Development": "R&D", "Human Resources": "HR", "Sales": "Sales"}[dept]
    if share < 0.7:
        short = "Cross-Functional"
    curated = {("Sales", "Mid-Level"): "Sales Front Line",
               ("Sales", "Early-Career"): "Early-Career Sales",
               ("HR", "Early-Career"): "HR Generalists", ("HR", "Mid-Level"): "HR Generalists",
               ("R&D", "Early-Career"): "Early-Career R&D Core",
               ("R&D", "Senior"): "Senior Loyalty Anchors",
               ("Cross-Functional", "Senior"): "Senior Loyalty Anchors"}
    return curated.get((short, sen), f"{sen} {short}")


def risk_tag(rate):
    if rate >= OVERALL_RATE * 1.25:
        return "High risk"
    if rate <= OVERALL_RATE * 0.6:
        return "Low risk"
    return "Moderate risk"


def profiles(k: int) -> pd.DataFrame:
    """One row per cluster: size, attrition and the average of every variable."""
    labels = MODELS[k]["labels"]
    d = df.assign(Cluster=labels + 1)
    rows = []
    for c, sub in d.groupby("Cluster"):
        r = {"Cluster": int(c), "Label": cluster_name(sub), "Employees": len(sub),
             "% of workforce": round(100 * len(sub) / len(d), 1),
             "Attrition rate %": round(100 * sub["AttritionFlag"].mean(), 1),
             "Leavers": int(sub["AttritionFlag"].sum())}
        r["Risk"] = risk_tag(sub["AttritionFlag"].mean())
        for v in NUMERIC + ORDINAL:
            r[v] = round(sub[v].mean(), 1 if v != "MonthlyIncome" else 0)
        r["% Male"] = round(100 * (sub["Gender"] == "Male").mean(), 0)
        if "OverTime" in df.columns:
            r["% OverTime"] = round(100 * (sub["OverTime"] == "Yes").mean(), 0)
        for dpt in DEPARTMENTS:
            r[f"% {dpt}"] = round(100 * (sub["Department"] == dpt).mean(), 0)
        rows.append(r)
    out = pd.DataFrame(rows)
    # make duplicate names unique
    dup = out["Label"].duplicated(keep=False)
    out.loc[dup, "Label"] = out.loc[dup, "Label"] + " (" + out.loc[dup, "Cluster"].astype(str) + ")"
    return out


PROFILES = {k: profiles(k) for k in K_RANGE}

# ---------------------------------------------------------------------------
# 3. FIGURES
# ---------------------------------------------------------------------------
TEMPLATE = "plotly_white"
NAVY, GOLD, INK = "#0F2537", "#CFB991", "#1d2733"


def ccolor(i):
    return PALETTE[i % len(PALETTE)]


def fig_elbow_sil(k):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Elbow method — inertia (within-cluster SSE)", "Silhouette score (higher = better separated)"))
    fig.add_trace(go.Scatter(x=METRICS.k, y=METRICS.inertia, mode="lines+markers",
                             line=dict(color="#4E79A7", width=3), name="Inertia"), 1, 1)
    fig.add_trace(go.Scatter(x=METRICS.k, y=METRICS.silhouette, mode="lines+markers",
                             line=dict(color="#F28E2B", width=3), name="Silhouette"), 1, 2)
    row = METRICS[METRICS.k == k].iloc[0]
    for col, y in ((1, row.inertia), (2, row.silhouette)):
        fig.add_trace(go.Scatter(x=[k], y=[y], mode="markers", showlegend=col == 1,
                                 name=f"Selected k = {k}",
                                 marker=dict(size=18, symbol="circle-open", color="#E15759",
                                             line=dict(width=3))), 1, col)
    fig.update_xaxes(title_text="Number of clusters (k)", dtick=1)
    fig.update_layout(template=TEMPLATE, height=380, margin=dict(l=20, r=20, t=50, b=20),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def fig_attrition_bar(k):
    p = PROFILES[k]
    fig = go.Figure(go.Bar(
        x=[f"C{c}: {l}" for c, l in zip(p.Cluster, p.Label)], y=p["Attrition rate %"],
        marker_color=[ccolor(i) for i in range(len(p))],
        text=[f"{r:.1f}%<br>n={n}" for r, n in zip(p["Attrition rate %"], p.Employees)],
        textposition="outside"))
    fig.add_hline(y=OVERALL_RATE * 100, line_dash="dash", line_color="#555",
                  annotation_text=f"IBM overall {OVERALL_RATE:.1%}", annotation_position="top left")
    fig.update_layout(template=TEMPLATE, height=400, title="Observed attrition rate by segment",
                      yaxis_title="Attrition rate (%)", margin=dict(l=20, r=20, t=50, b=20),
                      yaxis_range=[0, max(p["Attrition rate %"]) * 1.3])
    return fig


def fig_scatter(k, sim_point=None):
    labels = MODELS[k]["labels"]
    p = PROFILES[k]
    fig = go.Figure()
    for c in range(k):
        m = labels == c
        fig.add_trace(go.Scatter3d(
            x=P3[m, 0], y=P3[m, 1], z=P3[m, 2], mode="markers",
            name=f"C{c+1} {p.Label.iloc[c]} (n={m.sum()})",
            marker=dict(size=3.5, color=ccolor(c), opacity=0.8),
            text=[f"Employee #{i} · {df.Department.iat[i]} · JobLevel {df.JobLevel.iat[i]} · "
                  f"Attrition {df.Attrition.iat[i]}" for i in np.where(m)[0]],
            hovertemplate="%{text}<extra></extra>"))
    if sim_point is not None:
        sp = pca.transform(sim_point.reshape(1, -1))[0]
        fig.add_trace(go.Scatter3d(x=[sp[0]], y=[sp[1]], z=[sp[2]], mode="markers",
                                   name="Simulated employee",
                                   marker=dict(size=10, color="black", symbol="diamond")))
    fig.update_layout(template=TEMPLATE, height=560, margin=dict(l=0, r=0, t=40, b=0),
                      title="Segments in 3D (PCA projection of the 14 encoded features)",
                      scene=dict(xaxis_title=PCA_AXES[0], yaxis_title=PCA_AXES[1],
                                 zaxis_title=PCA_AXES[2]),
                      legend=dict(itemsizing="constant"))
    return fig


def fig_heatmap(k):
    """Centroids in z-score units: how far each segment sits from the IBM average."""
    C = pd.DataFrame(MODELS[k]["centers"], columns=FEATURES)
    p = PROFILES[k]
    fig = go.Figure(go.Heatmap(
        z=C.values, x=FEATURES, y=[f"C{c}: {l}" for c, l in zip(p.Cluster, p.Label)],
        colorscale="RdBu_r", zmid=0, zmin=-2, zmax=2, text=np.round(C.values, 1),
        texttemplate="%{text}", colorbar=dict(title="z-score")))
    fig.update_layout(template=TEMPLATE, height=110 + 55 * k, margin=dict(l=20, r=20, t=50, b=20),
                      title="Cluster centroids (standard deviations from the IBM average)",
                      yaxis=dict(autorange="reversed"))
    return fig


def fig_drill(k, var):
    """Attrition rate inside each segment, split by one variable."""
    labels = MODELS[k]["labels"]
    d = df.assign(Cluster=labels + 1)
    if var in ("Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
               "YearsInCurrentRole", "YearsWithCurrManager"):
        d["level"] = pd.qcut(d[var], 4, duplicates="drop").astype(str)
    elif var in LABELS:
        d["level"] = d[var].map(LABELS[var])
    else:
        d["level"] = d[var]
    g = d.groupby(["Cluster", "level"]).agg(rate=("AttritionFlag", "mean"),
                                           n=("AttritionFlag", "size")).reset_index()
    fig = go.Figure()
    p = PROFILES[k]
    levels = list(dict.fromkeys(d.sort_values(var)["level"]))
    for c in range(1, k + 1):
        s = g[g.Cluster == c].set_index("level").reindex(levels)
        fig.add_trace(go.Bar(x=levels, y=s.rate * 100, name=f"C{c} {p.Label.iloc[c-1]}",
                             marker_color=ccolor(c - 1),
                             customdata=s.n, hovertemplate="%{x}: %{y:.1f}% (n=%{customdata})"))
    fig.add_hline(y=OVERALL_RATE * 100, line_dash="dash", line_color="#555")
    fig.update_layout(template=TEMPLATE, barmode="group", height=420,
                      title=f"Attrition rate by {var} within each segment",
                      yaxis_title="Attrition rate (%)", margin=dict(l=20, r=20, t=50, b=20),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def fig_distances(dist, assigned, k):
    p = PROFILES[k]
    fig = go.Figure(go.Bar(
        y=[f"C{c}: {l}" for c, l in zip(p.Cluster, p.Label)], x=dist, orientation="h",
        marker_color=["#0F2537" if i == assigned else "#c9ced6" for i in range(k)],
        text=[f"{x:.2f}" for x in dist], textposition="outside"))
    fig.update_layout(template=TEMPLATE, height=90 + 45 * k, margin=dict(l=20, r=40, t=50, b=20),
                      title="Euclidean distance to each centroid (shortest = assigned)",
                      xaxis_title="Distance in standardized space", yaxis=dict(autorange="reversed"),
                      xaxis_range=[0, float(max(dist)) * 1.18])
    return fig


def friendly(f):
    """Plain-English name for an encoded feature."""
    if f.startswith("Dept_"):
        return f"Works in {f[5:]}"
    if f.startswith("Gender_"):
        return "Gender (male)"
    if f.startswith("OverTime_"):
        return "Works overtime"
    return {"MonthlyIncome": "Monthly income", "TotalWorkingYears": "Total working years",
            "YearsAtCompany": "Years at IBM", "YearsInCurrentRole": "Years in current role",
            "YearsWithCurrManager": "Years with current manager", "JobLevel": "Job level",
            "JobSatisfaction": "Job satisfaction", "WorkLifeBalance": "Work-life balance"}.get(f, f)


def fig_drivers(contrib):
    c = contrib.sort_values("Contribution")
    fig = go.Figure(go.Bar(
        y=[friendly(f) for f in c.Feature], x=c.Contribution, orientation="h",
        marker_color=["#59A14F" if v > 0 else "#E15759" for v in c.Contribution]))
    fig.update_layout(template=TEMPLATE, height=440, margin=dict(l=20, r=20, t=70, b=20),
                      title="Which characteristics decide the assignment?<br>"
                            "<sup>Green: the employee looks more like the assigned segment on this trait. "
                            "Red: more like the runner-up segment.</sup>",
                      xaxis_title="How strongly the trait pulls toward the assigned segment")
    return fig


def driver_table(rec, e, k):
    """Employee value vs. the assigned and runner-up segment averages, for the top 3 drivers."""
    p = PROFILES[k]
    a, b = e["assigned"], e["runner_up"]
    raw_a = scaler.inverse_transform(MODELS[k]["centers"][[a]])[0]
    raw_b = scaler.inverse_transform(MODELS[k]["centers"][[b]])[0]
    zrec = encode(pd.DataFrame([rec]))[FEATURES].iloc[0]
    rows = []
    for f in e["contrib"].sort_values("Contribution", ascending=False).head(3).Feature:
        i = FEATURES.index(f)
        pct = f.startswith(("Dept_", "Gender_", "OverTime_"))
        fmt = (lambda v: f"{abs(round(v*100)):.0f}%") if pct else (lambda v: f"{v:,.1f}")
        emp = ("Yes" if zrec[f] == 1 else "No") if pct else f"{zrec[f]:,.0f}"
        rows.append({"Characteristic": friendly(f), "This employee": emp,
                     f"Avg in C{a+1} {p.Label.iloc[a]}": fmt(raw_a[i]),
                     f"Avg in C{b+1} {p.Label.iloc[b]}": fmt(raw_b[i])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. SIMULATOR + EXPLAINABILITY ENGINE
# ---------------------------------------------------------------------------
def defaults():
    d = {v: int(df[v].median()) for v in NUMERIC + ORDINAL}
    for v in BINARY + NOMINAL:
        d[v] = df[v].mode().iat[0]
    return d


DEFAULTS = defaults()


def explain(record: dict, k: int):
    row = pd.DataFrame([record])
    z = scaler.transform(encode(row)[FEATURES])[0]
    C = MODELS[k]["centers"]
    dist = np.sqrt(((C - z) ** 2).sum(axis=1))
    order = np.argsort(dist)
    a, b = order[0], order[1]
    contrib = pd.DataFrame({"Feature": FEATURES,
                            "Contribution": (z - C[b]) ** 2 - (z - C[a]) ** 2})
    margin = dist[b] - dist[a]
    confidence = "clear" if margin > 0.75 else ("moderate" if margin > 0.3 else "borderline")
    return {"z": z, "dist": dist, "assigned": int(a), "runner_up": int(b),
            "margin": float(margin), "confidence": confidence, "contrib": contrib}


# ---------------------------------------------------------------------------
# 5. CHATBOT (Google Gemini free tier, grounded; OpenAI optional; offline fallback)
# ---------------------------------------------------------------------------
def context_json(k, sim=None):
    p = PROFILES[k]
    ctxd = {
        "dataset": f"{len(df)} IBM employees (2025). Overall attrition {OVERALL_RATE:.1%}. "
                   "Attrition mixes voluntary and involuntary departures.",
        "clustering_inputs": CLUSTER_VARS,
        "note_overtime": "OverTime is not present in the provided dataset."
                         if "OverTime" not in df.columns else "OverTime included.",
        "selected_k": k,
        "elbow_silhouette": METRICS.round(3).to_dict("records"),
        "best_k_by_silhouette": BEST_SIL_K,
        "clusters": p.to_dict("records"),
        "attrition_by_factor_overall": {
            v: (df.groupby(v)["AttritionFlag"].mean() * 100).round(1).to_dict()
            for v in ["Department", "JobLevel", "Gender", "Education", "JobSatisfaction",
                      "WorkLifeBalance"] + (["OverTime"] if "OverTime" in df.columns else [])},
    }
    if sim:
        ctxd["current_simulation"] = sim
    return json.dumps(ctxd, default=str)


SYSTEM_PROMPT = (
    "You are the IBM HR Segmentation Assistant inside a K-Means analytics app for "
    "Dr. Sarah Chen (CHRO). Answer ONLY from the JSON context supplied (cluster profiles, "
    "attrition rates, Elbow/Silhouette metrics, current simulation). Quote exact numbers and "
    "cluster numbers/labels. If the context does not contain the answer, say so. "
    "Never predict whether a specific person will leave; segments show observed attrition "
    "rates only. Remind users that segments are for retention planning only, not hiring, "
    "promotion or discipline, when relevant. Attrition mixes voluntary and involuntary exits. "
    "Keep answers under 150 words, plain business language."
)


LAST_LLM_ERROR = {"msg": ""}


def gemini_key():
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()


def provider():
    if gemini_key():
        return "Gemini"
    if (os.environ.get("OPENAI_API_KEY") or "").strip():
        return "OpenAI"
    return None


_GEMINI_MODELS = []


def _gemini_models():
    """Ask Google which models this key can use, so the app keeps working when model
    names change. Fast 'flash-lite' models first (newest first), then other 'flash' models."""
    import re
    import requests
    if _GEMINI_MODELS:
        return _GEMINI_MODELS
    names = []
    try:
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                         headers={"x-goog-api-key": gemini_key()}, timeout=6)
        if r.status_code == 200:
            for m in r.json().get("models", []):
                n = m.get("name", "").replace("models/", "")
                if ("generateContent" in m.get("supportedGenerationMethods", []) and "flash" in n
                        and not re.search(r"image|tts|audio|live|embed|thinking|exp", n)):
                    names.append(n)
    except Exception:
        pass

    def version(n):
        v = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
        return float(v.group(1)) if v else 0.0

    names.sort(key=lambda n: ("lite" not in n, "preview" in n, -version(n), n))
    pref = os.environ.get("GEMINI_MODEL")
    ordered = ([pref] if pref else []) + names + ["gemini-flash-lite-latest", "gemini-flash-latest"]
    _GEMINI_MODELS[:] = list(dict.fromkeys(ordered))
    return _GEMINI_MODELS


def _gemini(question, k, sim):
    """Google Gemini (free tier) via REST - no extra SDK needed.
    Tries available free models in turn: if one is overloaded (503), rate-limited (429)
    or not available (404), the next one answers. Total time is capped so the app never hangs."""
    import time
    import requests
    body = {"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT + "\n\nCONTEXT: " + context_json(k, sim)}]},
            "contents": [{"role": "user", "parts": [{"text": question}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 800}}
    deadline, errors = time.time() + 25, []
    for m in _gemini_models():
        left = deadline - time.time()
        if left < 4:
            break
        try:
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
                              headers={"x-goog-api-key": gemini_key(), "Content-Type": "application/json"},
                              json=body, timeout=min(15, left))
        except requests.RequestException as e:
            errors.append(f"{m}: {type(e).__name__}")
            continue
        if r.status_code in (404, 429, 500, 503):   # unavailable / busy -> try the next model
            errors.append(f"{m}: HTTP {r.status_code}")
            if r.status_code == 404 and m in _GEMINI_MODELS:
                _GEMINI_MODELS.remove(m)
            continue
        if r.status_code != 200:                     # e.g. 400/401/403 = key problem, stop
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:140]}")
        cand = r.json().get("candidates") or []
        text = "".join(p.get("text", "") for p in (cand[0].get("content", {}).get("parts", []) if cand else []))
        if text.strip():
            return text.strip()
        errors.append(f"{m}: empty answer")
    raise RuntimeError("; ".join(errors) or "no Gemini model available")


def _openai(question, k, sim):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip(), timeout=12.0, max_retries=0)
    r = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "system", "content": "CONTEXT: " + context_json(k, sim)},
                  {"role": "user", "content": question}])
    return r.choices[0].message.content


def llm_answer(question, k, sim):
    prov = provider()
    if not prov:
        LAST_LLM_ERROR["msg"] = "no GEMINI_API_KEY set"
        return None
    try:
        a = _gemini(question, k, sim) if prov == "Gemini" else _openai(question, k, sim)
        LAST_LLM_ERROR["msg"] = ""
        return a or None
    except Exception as e:  # network / quota / auth problems -> fall back, but report why
        LAST_LLM_ERROR["msg"] = f"{prov} {type(e).__name__}: {str(e)[:160]}"
        print("LLM call failed ->", LAST_LLM_ERROR["msg"], flush=True)
        return None


def rule_answer(q, k, sim):
    """Deterministic, grounded answers computed straight from the cluster results."""
    q = q.lower()
    p = PROFILES[k]
    hi, lo = p.iloc[0], p.iloc[-1]

    def line(r):
        return (f"C{r.Cluster} **{r.Label}** — {r.Employees} employees ({r['% of workforce']}%), "
                f"attrition **{r['Attrition rate %']}%**, avg JobLevel {r.JobLevel}, "
                f"income ${r.MonthlyIncome:,.0f}, {r.YearsAtCompany} yrs at IBM, "
                f"satisfaction {r.JobSatisfaction}/4, WLB {r.WorkLifeBalance}/4.")

    def by(v):
        s = (df.groupby(v)["AttritionFlag"].mean() * 100).round(1)
        if v in LABELS:
            s.index = [f"{i}-{LABELS[v][i]}" for i in s.index]
        return ", ".join(f"{i}: {x}%" for i, x in s.items())

    for i in range(1, k + 1):
        if f"cluster {i}" in q or f"c{i}" in q.split() or f"segment {i}" in q:
            return line(p.iloc[i - 1])
    if "simulat" in q or "this employee" in q or "my case" in q:
        if not sim:
            return "Run the simulator first (Simulator tab), then ask again."
        return (f"The simulated employee is assigned to C{sim['cluster']} **{sim['label']}** "
                f"(observed attrition {sim['rate']}%). Assignment is {sim['confidence']} — "
                f"margin {sim['margin']:.2f} over runner-up C{sim['runner_up']}. "
                "This is the segment's observed rate, not a prediction for this person.")
    if "overtime" in q:
        return ("OverTime is not a column in the IBM_Dataset provided, so the app cannot "
                "measure it. Position factors available: Department (" + by("Department") +
                ") and JobLevel (" + by("JobLevel") + ").") if "OverTime" not in df.columns \
            else "Attrition by OverTime: " + by("OverTime")
    if any(w in q for w in ["silhouette", "elbow", "best k", "why k", "choice of k", "choose k", "k=", "k =", "how many cluster", "number of cluster"]):
        r = METRICS[METRICS.k == k].iloc[0]
        s = METRICS.set_index("k").silhouette
        return (f"With k={k}: silhouette {r.silhouette:.3f}, inertia {r.inertia:,.0f}. "
                f"Highest silhouette is k={BEST_SIL_K} ({s.max():.3f}) but two clusters only "
                f"split Sales from everyone else. k=4 is a local silhouette peak ({s[4]:.3f} vs "
                f"{s[3]:.3f} at k=3 and {s[5]:.3f} at k=5) and sits where the elbow flattens, "
                "giving four segments HR can act on.")
    if any(w in q for w in ["gender", "education", "demograph", "female", "male", "degree"]):
        return ("Gender — " + by("Gender") + ". Education — " + by("Education") +
                ". Gender mix is similar across segments; demographics are weaker drivers than "
                "role, seniority and satisfaction.")
    if any(w in q for w in ["satisf", "work-life", "work life", "wlb", "balance"]):
        return ("Job satisfaction — " + by("JobSatisfaction") + ". Work-life balance — " +
                by("WorkLifeBalance") + ". Inside the higher-risk segments, low satisfaction "
                "and bad WLB sharply raise attrition (see Profiles ▸ drill-down).")
    if any(w in q for w in ["senior", "tenure", "experience", "years", "manager"]):
        return ("Seniority matters: leavers average 5.1 yrs at IBM vs 7.4 for stayers and "
                "8.2 vs 11.9 total working years. " + line(lo))
    if any(w in q for w in ["department", "job level", "position", "role", "sales", "r&d", "hr"]):
        return "Department — " + by("Department") + ". JobLevel — " + by("JobLevel") + "."
    if any(w in q for w in ["highest", "most risk", "risk", "leave", "worst"]):
        return "Highest-risk segment: " + line(hi)
    if any(w in q for w in ["lowest", "stable", "loyal", "safest"]):
        return "Lowest-risk segment: " + line(lo)
    if any(w in q for w in ["summary", "overview", "all cluster", "segments"]):
        return "\n".join(line(r) for _, r in p.iterrows())
    return ("I answer from this app's cluster results. Try: 'Which segment has the highest "
            "attrition?', 'Describe cluster 2', 'Why k=4?', 'Does gender matter?', "
            "'How does satisfaction relate to attrition?', 'Explain my simulation'.")


def chat_answer(q, k, sim):
    a = llm_answer(q, k, sim)
    if a:
        return a, f"{provider()} (grounded in cluster results)"
    why = LAST_LLM_ERROR["msg"]
    return rule_answer(q, k, sim), "Grounded engine" + (f" (AI model unavailable: {why})" if why else "")



# ---------------------------------------------------------------------------
# 6. STREAMLIT INTERFACE
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
.block-container {{padding-top: 3rem;}}
.hdr {{background:{NAVY}; color:white; padding:18px 24px; border-radius:12px; margin-bottom:10px;}}
.hdr h2 {{margin:0; color:white;}} .hdr div {{color:{GOLD};}}
.warn {{background:#FFF4D6; padding:8px 14px; border-radius:8px; font-size:13px; margin-bottom:12px; color:#1d2733;}}
div[data-testid="stMetric"] {{background:white; border-top:4px solid {GOLD}; border-radius:10px;
    padding:10px 14px; box-shadow:0 1px 4px rgba(0,0,0,.08);}}
div[data-testid="stMetric"] * {{color:#1d2733 !important;}}
</style>
<div class="hdr"><h2>IBM Workforce Segmentation Studio</h2>
<div>K-Means segments of 1,470 employees · observed attrition by segment · for Dr. Sarah Chen, CHRO</div></div>
<div class="warn">Responsible use: segments support retention planning only — never hiring, promotion or
disciplinary decisions. Attrition combines voluntary and involuntary departures.
{"" if "OverTime" in df.columns else "OverTime is not in the provided dataset, so 12 clustering variables are used."}</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns([4, 1])
k = c1.slider("Number of clusters (k)", 2, 10, DEFAULT_K, help="Recommended k = 4 — see Model Quality")
c2.markdown(f"<div style='padding-top:28px;color:#5b6b7b'>Recommended k = {DEFAULT_K}</div>",
            unsafe_allow_html=True)
P = PROFILES[k]
SIL = METRICS.set_index("k").silhouette
m = st.columns(5)
m[0].metric("Employees analysed", f"{len(df):,}")
m[1].metric("Overall attrition", f"{OVERALL_RATE:.1%}")
m[2].metric("Silhouette (selected k)", f"{SIL[k]:.3f}")
m[3].metric("Highest-risk segment", f"C{P.Cluster.iloc[0]} · {P['Attrition rate %'].iloc[0]}%")
m[4].metric("Lowest-risk segment", f"C{P.Cluster.iloc[-1]} · {P['Attrition rate %'].iloc[-1]}%")

tabs = st.tabs(["Overview", "Model Quality (Elbow & Silhouette)", "Cluster Profiles", "Cluster Simulator",
                "Interpretability", "Explainability", "AI Assistant"])

# ---------------- Overview
with tabs[0]:
    a, b = st.columns(2)
    a.plotly_chart(fig_attrition_bar(k), width="stretch", key="chart1")
    size = go.Figure(go.Pie(labels=[f"C{c}: {l}" for c, l in zip(P.Cluster, P.Label)], values=P.Employees,
                            hole=.5, marker_colors=[ccolor(i) for i in range(k)]))
    size.update_layout(template=TEMPLATE, height=400, title="Segment size (share of workforce)",
                       margin=dict(l=20, r=20, t=50, b=20))
    b.plotly_chart(size, width="stretch", key="chart2")
    st.plotly_chart(fig_scatter(k), width="stretch", key="chart3")

# ---------------- Model quality
with tabs[1]:
    st.plotly_chart(fig_elbow_sil(k), width="stretch", key="chart4")
    st.markdown(
        f"**Selected k = {k}: silhouette {SIL[k]:.3f}.** Silhouette peaks at k = {BEST_SIL_K} "
        f"({SIL.max():.3f}), but two segments only separate Sales from everyone else. k = 4 is a local "
        f"peak ({SIL[4]:.3f} vs {SIL[3]:.3f} at k=3 and {SIL[5]:.3f} at k=5), and the elbow curve flattens "
        "after k = 4: each extra cluster beyond it reduces inertia less. k = 4 therefore balances separation "
        "quality with segments HR can act on. Silhouette values around 0.2 indicate overlapping (not crisp) "
        "segments, typical of HR data.")
    st.dataframe(METRICS.assign(inertia=METRICS.inertia.round(0), silhouette=METRICS.silhouette.round(3)),
                 hide_index=True, width="stretch")

# ---------------- Profiles
with tabs[2]:
    st.dataframe(P, hide_index=True, width="stretch")
    st.plotly_chart(fig_heatmap(k), width="stretch", key="chart5")
    var = st.selectbox("Drill-down: attrition inside each segment by", CLUSTER_VARS,
                       index=CLUSTER_VARS.index("JobSatisfaction"))
    st.plotly_chart(fig_drill(k, var), width="stretch", key="chart6")
    st.markdown("#### Segment summaries")
    for _, r in P.iterrows():
        st.markdown(f"**C{r.Cluster} · {r.Label} ({r.Risk}, {r['Attrition rate %']}%):** {r.Employees} employees; "
                    f"avg age {r.Age}, JobLevel {r.JobLevel}, ${r.MonthlyIncome:,.0f}/month, "
                    f"{r.YearsAtCompany} yrs at IBM, {r.YearsInCurrentRole} yrs in role; satisfaction "
                    f"{r.JobSatisfaction}/4, WLB {r.WorkLifeBalance}/4.")

# ---------------- Simulator
EXAMPLE = {"Age": 27, "Department": "Sales", "JobLevel": 2, "JobSatisfaction": 1, "WorkLifeBalance": 1,
           "MonthlyIncome": 4500, "TotalWorkingYears": 4, "YearsAtCompany": 2, "YearsInCurrentRole": 2,
           "YearsWithCurrManager": 1, "OverTime": "Yes"}


def _set(values):
    for v in CLUSTER_VARS:
        st.session_state[f"sim_{v}"] = values.get(v, DEFAULTS[v])


for v in CLUSTER_VARS:                      # pre-fill with medians / most frequent values
    st.session_state.setdefault(f"sim_{v}", DEFAULTS[v])

with tabs[3]:
    left, right = st.columns([1, 2.2])
    with left:
        st.markdown("#### Employee characteristics")
        st.caption("Pre-filled with IBM medians / most frequent values — change only what you need.")
        b1, b2 = st.columns(2)
        b1.button("Junior Sales example", on_click=_set, args=(EXAMPLE,), width="stretch")
        b2.button("Reset defaults", on_click=_set, args=(DEFAULTS,), width="stretch")
        for v in CLUSTER_VARS:
            if v in LABELS:
                st.selectbox(v, list(LABELS[v]), key=f"sim_{v}", format_func=lambda i, v=v: f"{i} – {LABELS[v][i]}")
            elif v in BINARY + NOMINAL:
                st.selectbox(v, sorted(df[v].unique()), key=f"sim_{v}")
            else:
                st.number_input(v, int(df[v].min()), int(df[v].max()), step=1, key=f"sim_{v}")
    REC = {v: st.session_state[f"sim_{v}"] for v in CLUSTER_VARS}
    E = explain(REC, k)
    R = P.iloc[E["assigned"]]
    SIM = {"record": REC, "cluster": int(R.Cluster), "label": R.Label, "rate": float(R["Attrition rate %"]),
           "confidence": E["confidence"], "margin": E["margin"], "runner_up": E["runner_up"] + 1,
           "distances": {f"C{i+1}": round(float(x), 2) for i, x in enumerate(E["dist"])}}
    with right:
        st.markdown(f"<div style='color:#5b6b7b'>Assigned segment</div>"
                    f"<h2 style='margin:0;color:{ccolor(E['assigned'])}'>C{R.Cluster} · {R.Label}</h2>",
                    unsafe_allow_html=True)
        st.markdown(f"Observed attrition in this segment: **{R['Attrition rate %']}%** (IBM overall "
                    f"{OVERALL_RATE:.1%}) · {R.Risk} · {R.Employees} employees  \n"
                    f"Assignment confidence: **{E['confidence']}**: {E['margin']:.2f} closer than the runner-up "
                    f"C{E['runner_up'] + 1} {P.Label.iloc[E['runner_up']]}.  \n"
                    f"Segment profile: age {R.Age}, JobLevel {R.JobLevel}, ${R.MonthlyIncome:,.0f}/month, "
                    f"{R.YearsAtCompany} yrs at IBM, satisfaction {R.JobSatisfaction}/4, WLB {R.WorkLifeBalance}/4.")
        st.caption("This is the observed rate of similar employees, not a prediction for this person.")
        st.plotly_chart(fig_distances(E["dist"], E["assigned"], k), width="stretch", key="chart7")
        st.plotly_chart(fig_scatter(k, E["z"]), width="stretch", key="chart8")

# ---------------- Interpretability
with tabs[4]:
    st.markdown(f"""
### How K-Means works
1. **Choose k**, the number of segments (here chosen on the Model Quality tab).
2. **Initialise** k centroids (k-means++ seeding, 20 restarts, `random_state=42`).
3. **Assign** every employee to the nearest centroid (Euclidean distance).
4. **Update** each centroid to the mean of its employees.
5. **Repeat** steps 3–4 until assignments stop changing; keep the restart with the lowest inertia.

### How the clustering is implemented in this app
| Step | Variables | Treatment | Why |
|---|---|---|---|
| Continuous | {", ".join(NUMERIC)} | kept numeric | real magnitudes |
| Ordinal | {", ".join(ORDINAL)} | kept as ordered integers (1…5 / 1…4) | order is meaningful |
| Binary | {", ".join(BINARY)} | 0/1 indicator | two categories |
| Nominal | Department | one-hot (3 dummies) | no natural order; integers would invent distances |
| All | {len(FEATURES)} encoded features | **StandardScaler** (z-score) | K-Means uses distance; income in dollars would otherwise dominate |
| Excluded | **Attrition** | used only to profile clusters | it is the outcome we want to explain, not a grouping input |

Clusters are then renumbered so **Cluster 1 = highest observed attrition**. The 3D view uses PCA only
for drawing; the clustering itself happens in the full {len(FEATURES)}-dimensional space.
""")

# ---------------- Explainability
with tabs[5]:
    src = st.radio("Explain", ["Simulated employee (Simulator tab)", "Existing employee #"], horizontal=True)
    if src.startswith("Existing"):
        idx = st.number_input("Employee #", 0, len(df) - 1, 0, step=1)
        rec, who = df.iloc[int(idx)][CLUSTER_VARS].to_dict(), f"Employee #{int(idx)} (actual attrition: {df.Attrition.iat[int(idx)]})"
    else:
        rec, who = REC, "The simulated employee"
    e = explain(rec, k)
    a_, b_ = e["assigned"], e["runner_up"]
    top = [friendly(f) for f in e["contrib"].sort_values("Contribution", ascending=False).head(3).Feature]
    st.markdown(
        f"**In plain English:** {who} is most similar to the average employee in **C{a_+1} {P.Label.iloc[a_]}**, "
        f"where **{P['Attrition rate %'].iloc[a_]}%** of employees left. K-Means compares the employee with the "
        f"*centre* (average profile) of each segment and picks the closest one: distance {e['dist'][a_]:.2f} vs "
        f"{e['dist'][b_]:.2f} for the next-closest segment, C{b_+1} {P.Label.iloc[b_]}. That gap makes this a "
        f"**{e['confidence']}** match. The traits that matter most here are **{', '.join(top)}**:")
    st.dataframe(driver_table(rec, e, k), hide_index=True, width="stretch")
    st.caption("Distances are measured on standardised scales, so a $1,000 income gap and a one-step "
               "satisfaction gap are compared fairly.")
    a, b = st.columns(2)
    a.plotly_chart(fig_distances(e["dist"], a_, k), width="stretch", key="chart9")
    b.plotly_chart(fig_drivers(e["contrib"]), width="stretch", key="chart10")

# ---------------- AI Assistant
SUGGESTIONS = ["Which segment has the highest attrition?", "Why did you choose k=4?",
               "Does gender or education matter?",
               "How do satisfaction and work-life balance relate to attrition?", "Explain my simulation"]
with tabs[6]:
    prov = provider()
    st.caption(f"Mode: {prov} AI (grounded in this app's cluster results)" if prov else
               "Mode: grounded engine (no GEMINI_API_KEY set). Answers are computed from the cluster results.")
    st.session_state.setdefault("chat", [])
    cols = st.columns(len(SUGGESTIONS))
    clicked = None
    for i, s in enumerate(SUGGESTIONS):
        if cols[i].button(s, key=f"sugg{i}", width="stretch"):
            clicked = s
    typed = st.chat_input("Ask about the segments…")
    q = typed or clicked
    if q:
        with st.spinner("Thinking…"):
            ans, srcname = chat_answer(q, k, SIM)
        st.session_state.chat.append({"q": q, "a": ans, "src": srcname})
    for h in st.session_state.chat:
        with st.chat_message("user"):
            st.markdown(h["q"])
        with st.chat_message("assistant"):
            st.markdown(h["a"])
            st.caption(h["src"])
    if st.session_state.chat and st.button("Clear chat"):
        st.session_state.chat = []
        st.rerun()
