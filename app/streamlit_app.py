
import streamlit as st
import sys
from pathlib import Path

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.translation import load_translation_model, translate_text

st.set_page_config(
    page_title="SwahiliTranslate AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #F4F7FA;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #102A43;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #627D98;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .language-label {
        color: #102A43;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .stTextArea textarea {
        background-color: white;
        border: 1px solid #D9E2EC;
        border-radius: 12px;
        font-size: 16px;
        color: #102A43;
    }

    div.stButton > button {
        background-color: #087E8B;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 12px 25px;
        font-weight: 700;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #05616B;
        color: white;
        border: none;
    }

    .info-card {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #D9E2EC;
        text-align: center;
    }

    .info-number {
        font-size: 25px;
        font-weight: 800;
        color: #087E8B;
    }

    .info-label {
        font-size: 13px;
        color: #627D98;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "translation" not in st.session_state:
    st.session_state.translation = ""

if "source_text" not in st.session_state:
    st.session_state.source_text = ""

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🌍 SwahiliTranslate AI")
    st.caption("Neural Machine Translation")

    st.divider()

    st.markdown("### About the project")

    st.write(
        "SwahiliTranslate AI translates Kiswahili "
        "sentences into English using a fine-tuned "
        "NLLB neural machine translation model."
    )

    st.divider()

    st.markdown("### Model information")

    st.write("**Base model:** NLLB-200-distilled-600M")
    st.write("**Fine-tuning:** LoRA")
    st.write("**Source language:** Kiswahili")
    st.write("**Target language:** English")

    st.divider()

    st.markdown("### Evaluation results")

    st.metric("BLEU Score", "35.96")
    st.metric("chrF++ Score", "57.84")

    st.caption("Evaluated on 1,826 test sentences.")

    st.divider()

    if st.button("🗑️ Clear translation history"):
        st.session_state.history = []
        st.rerun()

# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🌍 SwahiliTranslate AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Breaking language barriers through artificial intelligence.'
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def get_model():
    return load_translation_model()

with st.spinner("Preparing your translation engine..."):
    model, tokenizer = get_model()

st.success("Translation engine ready!", icon="✅")

# --------------------------------------------------
# EXAMPLE SENTENCES
# --------------------------------------------------

st.markdown("### Try an example")

examples = {
    "Greeting": "Habari yako? Natumaini unaendelea vizuri.",
    "Education": "Ninapenda kujifunza lugha mpya.",
    "Appreciation": "Asante sana kwa msaada wako.",
    "Everyday life": "Leo ninaenda sokoni kununua chakula."
}

selected_example = st.selectbox(
    "Choose a sample Kiswahili sentence:",
    ["Select an example"] + list(examples.keys())
)

if selected_example != "Select an example":
    st.session_state.input_text = examples[selected_example]

# --------------------------------------------------
# TRANSLATION PANELS
# --------------------------------------------------

left, right = st.columns(2, gap="large")

with left:
    st.markdown(
        '<div class="language-label">🇰🇪 Kiswahili</div>',
        unsafe_allow_html=True
    )

    swahili_text = st.text_area(
        "Enter your text",
        key="input_text",
        height=220,
        placeholder="Andika sentensi yako ya Kiswahili hapa..."
    )

with right:
    st.markdown(
        '<div class="language-label">🇬🇧 English</div>',
        unsafe_allow_html=True
    )

    st.text_area(
        "Translation output",
        value=st.session_state.translation,
        height=220,
        disabled=True,
        placeholder="Your English translation will appear here..."
    )

# --------------------------------------------------
# TRANSLATION ACTIONS
# --------------------------------------------------

translate_col, clear_col = st.columns([3, 1])

with translate_col:
    translate_button = st.button(
        "✨ Translate to English",
        type="primary"
    )

with clear_col:
    clear_button = st.button("Clear")

if clear_button:
    st.session_state.input_text = ""
    st.session_state.translation = ""
    st.session_state.source_text = ""
    st.rerun()

if translate_button:

    if not swahili_text.strip():
        st.warning("Please enter some Kiswahili text first.")

    else:
        try:
            with st.spinner("Translating your sentence..."):

                result = translate_text(
                    swahili_text,
                    model,
                    tokenizer
                )

            st.session_state.translation = result
            st.session_state.source_text = swahili_text

            st.session_state.history.insert(
                0,
                {
                    "source": swahili_text,
                    "translation": result
                }
            )

            st.success("Translation completed successfully!")

            st.rerun()

        except Exception as error:
            st.error(f"Translation error: {error}")

# --------------------------------------------------
# COPY-FRIENDLY OUTPUT
# --------------------------------------------------

if st.session_state.translation:

    st.divider()

    st.markdown("### Your translated sentence")

    st.code(
        st.session_state.translation,
        language=None
    )

# --------------------------------------------------
# TRANSLATION HISTORY
# --------------------------------------------------

if st.session_state.history:

    st.divider()

    st.markdown("### 🕘 Recent translations")

    for index, item in enumerate(st.session_state.history):

        with st.expander(
            f"Translation {index + 1}: {item['source'][:55]}"
        ):
            st.markdown("**Kiswahili**")
            st.write(item["source"])

            st.markdown("**English**")
            st.write(item["translation"])

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    "<center style='color:#627D98;'>"
    "SwahiliTranslate AI | Powered by Neural Machine Translation"
    "</center>",
    unsafe_allow_html=True
)
