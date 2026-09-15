import streamlit as st
import sys
import json
import pandas as pd
sys.path.insert(0, 'scripts')
from agent_04 import run_agent

st.set_page_config(
    page_title="Amazon Support Agent",
    page_icon="📦",
    layout="wide"
)

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #FF9900;
    text-align: center;
}
.sub-header {
    font-size: 1rem;
    color: #aaa;
    text-align: center;
    margin-bottom: 1rem;
}
.reply-box {
    background-color: #1e3a5f;
    color: #ffffff;
    padding: 1.2rem;
    border-radius: 10px;
    border-left: 5px solid #4da6ff;
    font-size: 1.05rem;
    line-height: 1.6;
    margin: 0.5rem 0;
    white-space: pre-wrap;
}
.escalate-box {
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #ff4444;
    background-color: #3d1a1a;
    color: #ffcccc;
    margin: 0.5rem 0;
}
.auto-box {
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #00cc44;
    background-color: #1a3d1a;
    color: #ccffcc;
    margin: 0.5rem 0;
}
.metric-card {
    background-color: #1e1e2e;
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid #333;
}
</style>
""", unsafe_allow_html=True)

INTENT_COLORS = {
    'ORDER_STATUS':       '#FF9900',
    'DELIVERY_ISSUE':     '#ff4444',
    'RETURN_REFUND':      '#4da6ff',
    'PAYMENT_ISSUE':      '#cc44ff',
    'TECHNICAL_ISSUE':    '#44ff88',
    'PRODUCT_ISSUE':      '#ffaa44',
    'CUSTOMER_COMPLAINT': '#ff44aa',
}
INTENT_EMOJI = {
    'ORDER_STATUS':       '📦',
    'DELIVERY_ISSUE':     '🚚',
    'RETURN_REFUND':      '↩️',
    'PAYMENT_ISSUE':      '💳',
    'TECHNICAL_ISSUE':    '🔧',
    'PRODUCT_ISSUE':      '⚠️',
    'CUSTOMER_COMPLAINT': '😤',
}

EXAMPLES = {
    "📦 Order Status":    "Where is my order #114-5647382? It was supposed to arrive 3 days ago!",
    "🚚 Delivery Issue":  "My package was delivered to the wrong address. The courier left it at my neighbour's house.",
    "↩️ Return Request":  "I want to return an item I bought last week. It doesn't fit. How do I get a refund?",
    "💳 Payment Issue":   "I was charged twice for the same order. Please refund the extra charge immediately!",
    "🔧 Technical Issue": "Amazon Prime Video won't load on my Fire TV Stick. It just shows a black screen.",
    "⚠️ Product Issue":   "The laptop I received is completely broken. Screen is cracked and won't turn on.",
    "😤 Complaint":       "This is absolutely ridiculous! Third time my order has been delayed. Terrible service!",
    "⚖️ Legal Threat":    "I am calling my lawyer tomorrow. Amazon committed fraud and I will sue you in court!",
}

# Initialize session state
if 'message' not in st.session_state:
    st.session_state.message = ''
if 'result' not in st.session_state:
    st.session_state.result = None

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📋 Try Example Messages")
    st.markdown("Click any to load it:")

    for label, msg in EXAMPLES.items():
        if st.button(label, use_container_width=True, key=label):
            st.session_state.message = msg
            st.session_state.result = None
            st.rerun()

    st.divider()
    st.markdown("### 🤖 Model Info")
    st.info(
        "**Model:** openai/gpt-oss-20b\n\n"
        "**Provider:** Groq API\n\n"
        "**Retrieval:** Keyword RAG\n\n"
        "**Dataset:** AmazonHelp Twitter"
    )
    st.divider()
    st.markdown("### 🎯 Evaluation")
    try:
        with open('results/metrics.json') as f:
            m = json.load(f)
        st.metric("Accuracy", f"{m['agent_accuracy']*100:.1f}%",
                  f"+{m['improvement_over_trivial']*100:.1f}% vs trivial")
        st.metric("Reply Quality", "4.44 / 5.0")
    except:
        st.metric("Accuracy", "67.7%", "+39.1% vs trivial")
        st.metric("Reply Quality", "4.44 / 5.0")

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-header">📦 Amazon Support Agent</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered customer support · '
            'Hiver SDE Intern Assignment · by Subikshan M</div>',
            unsafe_allow_html=True)
st.divider()

# ============================================================
# INPUT
# ============================================================
st.markdown("#### 💬 Customer Message")

# Show current message in text area
user_input = st.text_area(
    label="Enter message:",
    value=st.session_state.message,
    height=130,
    placeholder="Type a customer support message or click an example on the left...",
    label_visibility="collapsed"
)

# Update session state when user types
if user_input != st.session_state.message:
    st.session_state.message = user_input
    st.session_state.result = None

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    analyze_btn = st.button("🚀 Analyze", type="primary",
                            use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.session_state.message = ''
    st.session_state.result = None
    st.rerun()

# ============================================================
# RUN AGENT
# ============================================================
if analyze_btn:
    if not st.session_state.message.strip():
        st.warning("⚠️ Please enter a message first!")
    else:
        with st.spinner("🤖 Analyzing..."):
            try:
                st.session_state.result = run_agent(
                    st.session_state.message.strip()
                )
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ============================================================
# SHOW RESULTS
# ============================================================
if st.session_state.result:
    result = st.session_state.result
    st.divider()
    st.markdown("## 🔍 Analysis Results")

    col_a, col_b = st.columns(2)

    # Intent box
    with col_a:
        intent = result['intent']
        conf   = result['intent_confidence']
        color  = INTENT_COLORS.get(intent, '#FF9900')
        emoji  = INTENT_EMOJI.get(intent, '❓')
        reason = result.get('intent_reasoning', '')

        st.markdown(
            f"<div style='background:{color}22;"
            f"border-left:5px solid {color};"
            f"padding:1rem;border-radius:10px'>"
            f"<h3 style='color:{color};margin:0'>{emoji} {intent}</h3>"
            f"<p style='color:#ccc;margin:0.3rem 0 0 0'>"
            f"Confidence: <b>{conf*100:.0f}%</b></p>"
            f"<p style='color:#aaa;font-size:0.9rem;"
            f"margin:0.3rem 0 0 0'>{reason}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

    # Escalation box
    with col_b:
        action = result['escalation_action']
        ereason = result['escalation_reason']

        if action == 'ESCALATE':
            st.markdown(
                f"<div class='escalate-box'>"
                f"<h3 style='margin:0;color:#ff6666'>🚨 ESCALATE TO HUMAN</h3>"
                f"<p style='margin:0.4rem 0 0 0'>{ereason}</p>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='auto-box'>"
                f"<h3 style='margin:0;color:#66ff88'>✅ AUTO HANDLE</h3>"
                f"<p style='margin:0.4rem 0 0 0'>{ereason}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

    # Confidence bar
    st.markdown("#### 📊 Confidence")
    st.progress(float(conf))
    st.caption(f"Confidence: {conf*100:.0f}% | "
               f"Examples retrieved: {result['similar_examples_used']}")

    # Draft reply - visible on dark and light mode
    st.markdown("#### 💬 Suggested Reply")
    draft = result['draft_reply']

    # Use st.success/info for always-visible colored box
    st.info(draft)

    char_count = len(draft)
    if char_count <= 280:
        st.success(f"✅ {char_count}/280 characters — fits in a tweet!")
    else:
        st.warning(f"⚠️ {char_count}/280 characters — slightly over Twitter limit")

    with st.expander("🔧 Raw JSON Output"):
        st.json(result)

# ============================================================
# DASHBOARD
# ============================================================
st.divider()

with st.expander("📊 Evaluation Results Dashboard"):
    st.markdown("### Accuracy vs Baselines")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='metric-card'>
        <h2 style='color:#ff4444'>28.6%</h2>
        <p style='color:#aaa'>Trivial Baseline</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='metric-card'>
        <h2 style='color:#FF9900'>42.3%</h2>
        <p style='color:#aaa'>Keyword Baseline</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='metric-card'>
        <h2 style='color:#00cc44'>67.7%</h2>
        <p style='color:#aaa'>Our Agent ✨</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Per-Intent Accuracy")
    chart_data = pd.DataFrame({
        'Agent %':   [86.7, 86.7, 80.6, 63.0, 59.1, 55.6, 54.5],
        'Keyword %': [33.3, 40.0,  2.8, 85.2, 40.9, 41.7,  0.0],
    }, index=['PAYMENT','RETURN','DELIVERY','COMPLAINT',
              'TECHNICAL','ORDER','PRODUCT'])
    st.bar_chart(chart_data)

    st.markdown("### LLM Judge Scores (out of 5.0)")
    judge_data = pd.DataFrame({
        'Score': [4.14, 4.28, 4.17, 4.76, 4.86, 4.44]
    }, index=['Relevance','Tone','Actionability',
              'Accuracy','Conciseness','Overall'])
    st.bar_chart(judge_data)

    try:
        rdf = pd.read_csv('results/evaluation_results.csv')
        st.markdown("### Sample Results")
        st.dataframe(
            rdf[['message','true_intent','agent_intent',
                 'agent_correct']].head(20),
            use_container_width=True
        )
    except:
        pass

# ============================================================
# FAILURE ANALYSIS
# ============================================================
with st.expander("🔍 Failure Analysis — Top 5 Modes"):
    data = [
        ("1. ORDER_STATUS ↔ DELIVERY_ISSUE", "9 cases",
         "@AmazonHelp No I contacted just delayed 😑",
         "'Delayed' triggers delivery keywords but customer wants status update."),
        ("2. COMPLAINT ↔ TECHNICAL_ISSUE", "8 cases",
         "None of your links work. Most pathetic service ever",
         "Technical words + frustration confuse the classifier."),
        ("3. Short / Vague Messages", "17 cases (27.9%)",
         "@AmazonHelp ?? 😬",
         "Under 80 chars = insufficient context for classification."),
        ("4. Non-English Messages", "Observed",
         "@AmazonHelp Oui si je viens ici... (French)",
         "ASCII filter lets some non-English through. No language detection."),
        ("5. Sarcasm", "Observed",
         "Definitely think Amazon packed this efficiently... (sarcastic)",
         "Sarcastic praise read as positive. No sarcasm detection."),
    ]
    for mode, count, example, hyp in data:
        st.markdown(f"**{mode}** — `{count}`")
        st.code(example)
        st.markdown(f"💡 {hyp}")
        st.divider()

# Footer
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.8rem'>"
    "Subikshan M | Hiver SDE Intern | AmazonHelp Twitter | Groq API"
    "</div>",
    unsafe_allow_html=True
)