import joblib
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Ticket Triage", layout="wide")

PRIORITY_RELIABLE_AT = 0.6


@st.cache_resource
def load_models():
    embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
    embedder.max_seq_length = 256
    return embedder, joblib.load('clf_type.joblib'), joblib.load('clf_priority.joblib')


embedder, clf_type, clf_priority = load_models()


def triage(texts, type_threshold):
    emb = embedder.encode(texts, normalize_embeddings=True, batch_size=32)
    tp = clf_type.predict_proba(emb)
    pp = clf_priority.predict_proba(emb)
    out = pd.DataFrame({
        'text': texts,
        'type': clf_type.classes_[tp.argmax(axis=1)],
        'type_conf': tp.max(axis=1).round(3),
        'priority': clf_priority.classes_[pp.argmax(axis=1)],
        'priority_conf': pp.max(axis=1).round(3),
    })
    out['priority_reliable'] = out['priority_conf'] >= PRIORITY_RELIABLE_AT
    out['needs_review'] = out['type_conf'] < type_threshold
    return out


st.title("AI Customer Support Ticket Triage")
st.caption("Predicts ticket type and priority. Low-confidence tickets are sent to human review.")

type_threshold = st.sidebar.slider("Type confidence threshold", 0.5, 0.9, 0.7, 0.05)
st.sidebar.write("Tickets with Type confidence below this go to human review.")

if 'review_queue' not in st.session_state:
    st.session_state.review_queue = []

tab1, tab2, tab3, tab4 = st.tabs(["Single ticket", "Batch CSV", "Review queue", "Model report"])

with tab1:
    text = st.text_area("Ticket text (subject + body)", height=180)
    if st.button("Analyze") and text.strip():
        r = triage([text], type_threshold).iloc[0]
        c1, c2 = st.columns(2)
        c1.metric("Type", r['type'])
        c1.caption(f"confidence {r['type_conf']:.2f}")
        c2.metric("Priority", r['priority'])
        c2.caption(f"confidence {r['priority_conf']:.2f}")
        if r['needs_review']:
            st.warning("Low confidence: sent to human review.")
            st.session_state.review_queue.append(r.to_dict())
        else:
            st.success(f"Confident: auto-route to the {r['type']} queue.")
        if not r['priority_reliable']:
            st.info("Priority confidence is low, treat the priority as a suggestion only.")

with tab2:
    file = st.file_uploader("Upload a CSV with a ticket text column", type="csv")
    if file:
        df = pd.read_csv(file)
        col = st.selectbox("Text column", df.columns)
        df = df.dropna(subset=[col]).head(500)
        st.caption(f"{len(df)} tickets will be processed (demo limit: 500)")
        if st.button("Run triage"):
            res = triage(df[col].astype(str).tolist(), type_threshold)
            st.write(f"Auto-routed: {(~res['needs_review']).sum()} | Human review: {res['needs_review'].sum()}")
            st.dataframe(res)
            st.download_button("Download results", res.to_csv(index=False), "triage_results.csv")
            st.session_state.review_queue.extend(res[res['needs_review']].to_dict('records'))

with tab3:
    q = pd.DataFrame(st.session_state.review_queue)
    st.write(f"{len(q)} tickets waiting for human review (this session)")
    if len(q) > 0:
        st.dataframe(q)
        st.download_button("Download review queue", q.to_csv(index=False), "review_queue.csv")

with tab4:
    st.subheader("Test results (group-aware split)")
    st.dataframe(pd.DataFrame({
        'model': ['Baseline', 'Logistic Regression', 'LinearSVC', 'Random Forest'],
        'Type accuracy': [0.399, 0.745, 0.774, 0.765],
        'Type macro-F1': [0.143, 0.764, 0.773, 0.739],
        'Priority accuracy': [0.424, 0.416, 0.434, 0.469],
        'Priority macro-F1': [0.198, 0.408, 0.408, 0.393],
    }), hide_index=True)
    st.subheader("Type: confidence threshold vs accuracy")
    st.dataframe(pd.DataFrame({
        'threshold': [0.0, 0.4, 0.5, 0.6, 0.7, 0.8],
        'auto-handled': ['100%', '99%', '95%', '67%', '47%', '35%'],
        'accuracy on auto-handled': [0.745, 0.749, 0.763, 0.841, 0.930, 0.987],
    }), hide_index=True)
    st.caption("Priority is only weakly predictable from ticket text, so review flags use Type confidence only.")