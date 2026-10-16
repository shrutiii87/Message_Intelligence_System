"""
app.py - Streamlit front-end for the Message Intelligence System.

Run locally:   streamlit run app.py
All model logic (training, evaluation, prediction) lives in model.py.
"""

import pandas as pd
import streamlit as st

from model import (
    DAY_NAMES,
    FEATURE_LABELS,
    MODEL_NAMES,
    RECOMMENDED_MODEL,
    explain_linear_svm,
    extract_text_features,
    predict_message,
    train_system,
)

st.set_page_config(
    page_title="Message Intelligence System",
    page_icon="📩",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Load / train once per server process
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Training models on the message dataset...")
def get_system():
    return train_system()


try:
    system = get_system()
except FileNotFoundError as err:
    st.error(f"{err}")
    st.stop()
except Exception as err:  # keep the page alive with a readable message
    st.error(f"Could not train the models: {err}")
    st.stop()


# --------------------------------------------------------------------------- #
# Session-state defaults + sample-message loader
# --------------------------------------------------------------------------- #
CUSTOM = "✏️ Write my own message"
DEFAULTS = {
    "example_choice": CUSTOM,
    "msg_text": "Can you send the report by end of day? Please confirm.",
    "activity": 55.0,
    "account_age": 300,
    "sent_24h": 6,
    "hour": 12,
    "day_name": "Wednesday",
}
for _key, _value in DEFAULTS.items():
    st.session_state.setdefault(_key, _value)

example_labels = {}
for _i, _row in system.examples.iterrows():
    _icon = "🚨 Spam" if int(_row["spam_label"]) == 1 else "✅ Legit"
    _text = _row["message_text"]
    _short = _text if len(_text) <= 70 else _text[:67] + "..."
    example_labels[f"{_icon} - {_short}"] = _i


def load_example():
    choice = st.session_state.example_choice
    if choice not in example_labels:
        return
    row = system.examples.iloc[example_labels[choice]]
    st.session_state.msg_text = str(row["message_text"])
    st.session_state.activity = float(min(max(row["sender_activity_score"], 5.0), 100.0))
    st.session_state.account_age = int(min(max(row["sender_account_age_days"], 1), 1500))
    st.session_state.sent_24h = int(min(max(row["messages_sent_last_24h"], 0), 40))
    st.session_state.hour = int(row["hour_of_day"])
    st.session_state.day_name = DAY_NAMES[int(row["day_of_week"])]


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "Classifier",
        MODEL_NAMES,
        index=MODEL_NAMES.index(RECOMMENDED_MODEL),
        format_func=lambda n: f"{n}  ⭐ recommended" if n == RECOMMENDED_MODEL else n,
        help="The final report recommends the Linear SVM for deployment.",
    )
    scores = system.metrics.loc[model_name]
    st.markdown(f"**Test-set performance** ({system.n_test:,} messages)")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{scores['Accuracy']:.2%}")
    c2.metric("F1 score", f"{scores['F1 Score']:.2%}")
    c1.metric("Precision", f"{scores['Precision']:.2%}")
    c2.metric("Recall", f"{scores['Recall']:.2%}")

    st.divider()
    st.caption(
        f"Trained on {system.n_train:,} of {system.n_total:,} messages "
        f"({system.spam_rate:.1%} spam). Median imputation + StandardScaler are "
        "fitted on the training split only."
    )


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("📩 Message Intelligence System")
st.markdown(
    "Classify a message as **spam** or **legitimate (ham)** using its text statistics, "
    "sender behaviour and timing - with KNN, SVM and Naive Bayes classifiers."
)

tab_check, tab_compare, tab_about = st.tabs(["🔍 Check a message", "📊 Model comparison", "ℹ️ About"])


# --------------------------------------------------------------------------- #
# Tab 1 - check a message
# --------------------------------------------------------------------------- #
with tab_check:
    st.selectbox(
        "Start from a real example from the dataset (or write your own)",
        [CUSTOM] + list(example_labels.keys()),
        key="example_choice",
        on_change=load_example,
    )

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Message")
        st.text_area(
            "Message text",
            key="msg_text",
            height=150,
            placeholder="Type or paste a message... (press Ctrl+Enter to apply)",
        )
        text_feats = extract_text_features(st.session_state.msg_text)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Length", text_feats["message_length"])
        m2.metric("Words", text_feats["word_count"])
        m3.metric("URL found", "Yes" if text_feats["num_urls"] else "No")
        m4.metric("Digits", text_feats["num_digits"])
        st.caption("These four values are computed automatically from the text.")

    with right:
        st.subheader("Sender & timing")
        st.slider("Sender activity score", 5.0, 100.0, step=0.1, key="activity")
        st.slider("Sender account age (days)", 1, 1500, step=1, key="account_age")
        st.slider("Messages sent in the last 24 hours", 0, 40, step=1, key="sent_24h")
        t1, t2 = st.columns(2)
        t1.slider("Hour of day", 0, 23, step=1, key="hour")
        t2.selectbox("Day of week", DAY_NAMES, key="day_name")

    features = {
        **text_feats,
        "sender_activity_score": st.session_state.activity,
        "sender_account_age_days": st.session_state.account_age,
        "messages_sent_last_24h": st.session_state.sent_24h,
        "hour_of_day": st.session_state.hour,
        "day_of_week": DAY_NAMES.index(st.session_state.day_name),
    }

    st.divider()
    st.subheader("Result")

    if not st.session_state.msg_text.strip():
        st.info("Enter a message above to see the prediction.")
    else:
        result = predict_message(system, model_name, features)
        spam_prob = min(max(result["spam_probability"], 0.0), 1.0)

        r_left, r_right = st.columns([2, 3], gap="large")
        with r_left:
            if result["label"] == 1:
                st.error(f"### 🚨 Spam\n{model_name} classifies this message as **spam**.")
            else:
                st.success(f"### ✅ Legitimate\n{model_name} classifies this message as **legitimate**.")
            st.progress(spam_prob, text=f"Spam probability: {spam_prob:.1%}")
            if result["borderline"]:
                st.info("This message sits right on the decision boundary - treat the result as uncertain.")
            if model_name == "Naive Bayes":
                st.caption(
                    "Naive Bayes probabilities are over-confident because its independence "
                    "assumption does not hold on this data (see the model comparison tab)."
                )
            elif model_name.startswith("SVM"):
                st.caption("SVM probabilities are sigmoid-calibrated estimates of the SVM score.")

            ranges = system.feature_ranges
            outside = [
                FEATURE_LABELS[name]
                for name in ("message_length", "word_count")
                if not (ranges.loc[name, "min"] <= features[name] <= ranges.loc[name, "max"])
            ]
            if outside:
                st.warning(
                    "Outside the range seen in training data: "
                    + ", ".join(outside)
                    + ". The prediction may be less reliable."
                )

        with r_right:
            st.markdown("**What every model says about this message**")
            rows = []
            for name in MODEL_NAMES:
                res = predict_message(system, name, features)
                rows.append(
                    {
                        "Model": name + (" ⭐" if name == RECOMMENDED_MODEL else ""),
                        "Verdict": res["label_name"],
                        "Spam probability": f"{res['spam_probability']:.1%}",
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True)

        with st.expander("🔎 Why? Feature contributions (Linear SVM)"):
            contrib = explain_linear_svm(system, features)
            st.bar_chart(contrib.set_index("Feature")["Contribution"], horizontal=True)
            st.caption(
                "Contribution = Linear SVM weight x the (scaled) feature value. "
                "Positive values push the message towards **spam**, negative values towards **legitimate**."
            )
            st.dataframe(contrib.round(3), hide_index=True)


# --------------------------------------------------------------------------- #
# Tab 2 - model comparison
# --------------------------------------------------------------------------- #
with tab_compare:
    st.subheader("KNN vs SVM vs Naive Bayes")
    st.caption(f"Evaluated on the held-out test split ({system.n_test:,} messages, stratified, random_state=42).")

    st.dataframe(system.metrics.round(4))

    best_f1 = system.metrics["F1 Score"].idxmax()
    st.success(
        f"Best F1 score: **{best_f1}** ({system.metrics.loc[best_f1, 'F1 Score']:.2%}). "
        "The final report recommends deploying the Linear SVM as the main filter."
    )

    st.subheader("Errors on the test set")
    errors = pd.DataFrame(
        {
            "Missed spam (false negatives)": [int(system.confusion[n][1, 0]) for n in MODEL_NAMES],
            "Legitimate blocked (false positives)": [int(system.confusion[n][0, 1]) for n in MODEL_NAMES],
        },
        index=MODEL_NAMES,
    )
    st.bar_chart(errors)

    st.subheader("Confusion matrices")
    cols = st.columns(len(MODEL_NAMES))
    for col, name in zip(cols, MODEL_NAMES):
        with col:
            st.markdown(f"**{name}**")
            cm = pd.DataFrame(
                system.confusion[name],
                index=["Actual ham", "Actual spam"],
                columns=["Pred. ham", "Pred. spam"],
            )
            st.dataframe(cm)

    st.markdown(
        """
**Reading the results**
- **SVM (Linear / RBF)** - fewest errors in both directions; the linear kernel is as accurate as RBF and far easier to explain.
- **KNN** - few false alarms but misses the most spam, and prediction slows as data grows.
- **Naive Bayes** - fast, but its independence and Gaussian assumptions do not hold here, so it flags many legitimate messages and is over-confident.
"""
    )


# --------------------------------------------------------------------------- #
# Tab 3 - about
# --------------------------------------------------------------------------- #
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
The **Message Intelligence System** is a supervised-learning spam filter built on a dataset of
5,200 messages. This app wraps the notebook pipeline so you can try it interactively.

**Pipeline**
1. 9 input features (message length, word count, URLs, digits, sender activity, account age,
   messages in 24h, hour of day, day of week)
2. Median imputation for missing sender values + `StandardScaler` (fitted on the training split only)
3. 80/20 stratified train/test split (`random_state=42`)
4. KNN (K=5), SVM (Linear), SVM (RBF) and Gaussian Naive Bayes
5. Evaluation with accuracy, precision, recall and F1

**Good to know**
- The keyword scores and special-character count are **not** used as inputs, so the models
  rely on length, URLs, digits, sender behaviour and timing.
- The dataset is highly repetitive, so live traffic may be harder than these test scores suggest.
- This is a learning project, not a production spam filter.
"""
    )

    st.subheader("Features used by the models")
    feature_table = system.feature_ranges.rename(columns={"min": "Min", "max": "Max", "median": "Median"}).copy()
    feature_table.index = [FEATURE_LABELS[i] for i in feature_table.index]
    st.dataframe(feature_table.round(2))