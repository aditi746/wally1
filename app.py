import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from kmodes.kprototypes import KPrototypes
from mlxtend.frequent_patterns import apriori, association_rules

# ───────────────────────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio Analytics Dashboard", page_icon="📈", layout="wide")

# ───────────────────────────────────────────────────────────────
# DATA LOADING
# ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path="IA_PBL_DA_MJ25GF015 (2).xlsx", sheet="streamlit_df"):
    return pd.read_excel(path, sheet_name=sheet)

df = load_data()

# ───────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ───────────────────────────────────────────────────────────────
st.sidebar.title("📂 Modules")
page = st.sidebar.radio(
    "Choose analytics module",
    ["📊 Descriptive Analytics",
     "🤖 Classification",
     "🎯 Clustering (K-Prototypes)",
     "🛒 Association Rules",
     "📈 Regression"]
)

# ───────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────────
def score_row(y_true, y_pred, name):
    return {"Model": name,
            "Accuracy": round(accuracy_score(y_true, y_pred), 3),
            "Precision": round(precision_score(y_true, y_pred, average='weighted'), 3),
            "Recall": round(recall_score(y_true, y_pred, average='weighted'), 3),
            "F1": round(f1_score(y_true, y_pred, average='weighted'), 3)}

def prettify_rules(rules_df):
    for col in ["antecedents", "consequents"]:
        rules_df[col] = rules_df[col].apply(lambda x: ", ".join(sorted(list(x))))
    return rules_df

# ───────────────────────────────────────────────────────────────
# MODULE 1: DESCRIPTIVE ANALYTICS
# ───────────────────────────────────────────────────────────────
if page == "📊 Descriptive Analytics":
    st.header("📊 Descriptive Portfolio Insights")
    with st.sidebar.expander("Filters", True):
        age_rng = st.slider("Age", int(df.Age.min()), int(df.Age.max()), (int(df.Age.min()), int(df.Age.max())))
        income_rng = st.slider("Annual Income", int(df['Annual Income'].min()), int(df['Annual Income'].max()),
                               (int(df['Annual Income'].min()), int(df['Annual Income'].max())))
        risk_levels = st.multiselect("Risk Tolerance", df['Risk Tolerance'].unique(), default=list(df['Risk Tolerance'].unique()))
        show_raw = st.checkbox("Show raw data")
    view = df[
        (df.Age.between(*age_rng)) &
        (df['Annual Income'].between(*income_rng)) &
        (df['Risk Tolerance'].isin(risk_levels))
    ]
    st.success(f"Filtered records: {len(view)}")
    if show_raw:
        st.dataframe(view.head())

    st.subheader("Portfolio Visualizations")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(view, x="Annual Income", nbins=20, title="Income Distribution")
        st.plotly_chart(fig)
    with c2:
        fig2 = px.histogram(view, x="Net worth", nbins=20, title="Net Worth Distribution")
        st.plotly_chart(fig2)

# ───────────────────────────────────────────────────────────────
# MODULE 2: CLASSIFICATION
# ───────────────────────────────────────────────────────────────
elif page == "🤖 Classification":
    st.header("🤖 Recommended Portfolio Classifier")
    y = df["Recommended Portfolio"]
    X = pd.get_dummies(df.drop(columns=["UserID", "Recommended Portfolio"]), drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_sc, X_test_sc = scaler.transform(X_train), scaler.transform(X_test)

    models = {
        "KNN":               KNeighborsClassifier(n_neighbors=7),
        "Decision Tree":     DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":     RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    scores = []
    for name, mdl in models.items():
        X_tr = X_train_sc if name == "KNN" else X_train
        X_te = X_test_sc if name == "KNN" else X_test
        mdl.fit(X_tr, y_train)
        y_pred = mdl.predict(X_te)
        scores.append(score_row(y_test, y_pred, name))

    st.dataframe(pd.DataFrame(scores).set_index("Model"))

# ───────────────────────────────────────────────────────────────
# MODULE 3: K-PROTOTYPES CLUSTERING
# ───────────────────────────────────────────────────────────────
elif page == "🎯 Clustering (K-Prototypes)":
    st.header("🎯 K-Prototypes Portfolio Segmentation")

    num_cols = ["Age", "Investment Horizon", "Annual Income", "Net worth", "Projected ROI 5years",
                "Portfolio Equity(%)", "Portfolio Bonds(%)", "Portfolio Cash(%)", 
                "Portfolio RealEstate(%)", "Portfolio Crypto(%)", "Historical Return (%)", "Portfolio Volatility"]
    cat_cols = [c for c in df.columns if c not in num_cols + ["UserID", "Cluster"]]

    df_clustering = df.copy()
    df_clustering[num_cols] = df_clustering[num_cols].fillna(df_clustering[num_cols].mean())
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_clustering[col] = le.fit_transform(df_clustering[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    X_num = scaler.fit_transform(df_clustering[num_cols])
    X_cat = df_clustering[cat_cols].to_numpy()
    X_mix = np.hstack([X_num, X_cat])
    cat_idx = list(range(X_num.shape[1], X_mix.shape[1]))

    k = st.slider("k (clusters)", 2, 10, 4)
    gamma_val = st.number_input("γ (numeric vs categorical weight)", 0.0, 10.0, 0.0, 0.1)
    gamma = None if gamma_val == 0.0 else gamma_val

    kp = KPrototypes(n_clusters=k, init="Huang", n_init=10, gamma=gamma, random_state=42)
    clusters = kp.fit_predict(X_mix, categorical=cat_idx)
    df["Cluster"] = clusters
    st.success(f"Clustering complete: {k} clusters assigned.")

    costs = []
    for ki in range(2, 11):
        km = KPrototypes(n_clusters=ki, init="Huang", n_init=3, random_state=42)
        km.fit_predict(X_mix, categorical=cat_idx)
        costs.append(km.cost_)
    fig_cost, ax_cost = plt.subplots()
    ax_cost.plot(range(2, 11), costs, marker='o')
    ax_cost.set(title="Cost vs k", xlabel="k", ylabel="Cost")
    st.pyplot(fig_cost)

    persona_num = df.groupby("Cluster")[num_cols].mean().round(1)
    persona_cat = df.groupby("Cluster")[cat_cols].agg(lambda s: s.mode().iloc[0])
    persona = pd.concat([persona_num, persona_cat], axis=1)
    st.subheader("📌 Cluster Personas")
    st.dataframe(persona)

    st.download_button("📥 Download Clustered Data", df.to_csv(index=False).encode("utf-8"), file_name="clustered_data.csv")

# ───────────────────────────────────────────────────────────────
# MODULE 4: ASSOCIATION RULES
# ───────────────────────────────────────────────────────────────
elif page == "🛒 Association Rules":
    st.header("🛒 Portfolio Allocation Associations (Apriori)")

    alloc_cols_all = ["Portfolio Equity(%)", "Portfolio Bonds(%)", "Portfolio Cash(%)",
                      "Portfolio RealEstate(%)", "Portfolio Crypto(%)"]
    selected_cols = st.multiselect("Select columns for rule mining", alloc_cols_all, default=alloc_cols_all)

    if selected_cols:
        basket = (df[selected_cols] > 20).astype(int)
        min_sup = st.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01)
        min_conf = st.slider("Minimum Confidence", 0.1, 0.9, 0.6, 0.05)
        min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.2, 0.1)

        if st.button("Run Apriori"):
            freq_sets = apriori(basket, min_support=min_sup, use_colnames=True)
            if freq_sets.empty:
                st.warning("⚠️ No frequent itemsets. Try lowering support.")
            else:
                rules = association_rules(freq_sets, metric="confidence", min_threshold=min_conf)
                rules = rules[rules["lift"] >= min_lift]
                if rules.empty:
                    st.warning("⚠️ No rules match the criteria.")
                else:
                    rules = prettify_rules(rules).sort_values("lift", ascending=False)
                    st.dataframe(
                        rules[["antecedents", "consequents", "support", "confidence", "lift"]]
                        .style.format({"support": "{:.3f}", "confidence": "{:.2f}", "lift": "{:.2f}"})
                    )
    else:
        st.warning("Select at least one column to run Apriori.")

# ───────────────────────────────────────────────────────────────
# MODULE 5: REGRESSION
# ───────────────────────────────────────────────────────────────
elif page == "📈 Regression":
    st.header("📈 Regression: Predict Historical Return or Volatility")
    target = st.selectbox("Target variable", ["Historical Return (%)", "Portfolio Volatility"])
    y = df[target]
    X = pd.get_dummies(df.drop(columns=["UserID", "Recommended Portfolio", "Historical Return (%)", "Portfolio Volatility"]), drop_first=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    regs = {"Linear": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.001),
            "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42)}
    
    results = []
    for name, model in regs.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        results.append({
            "Model": name,
            "R²": round(model.score(X_te, y_te), 3),
            "RMSE": int(np.sqrt(((y_te - preds) ** 2).mean())),
            "MAE": int(np.abs(y_te - preds).mean())
        })

    st.dataframe(pd.DataFrame(results).set_index("Model"))
