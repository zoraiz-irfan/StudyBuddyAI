"""
StudyBuddy AI — Study Abroad & Visa Assistant
A Streamlit chatbot powered by the Groq API.
"""

import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from groq import Groq, APIError, APIConnectionError, RateLimitError

load_dotenv()

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="StudyBuddy AI | Study Abroad & Visa Assistant",
    page_icon="🎓",
    # layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Available models (see console.groq.com/docs/models for the latest list)
# ----------------------------------------------------------------------
MODELS = {
    "GPT-OSS 120B (best quality, recommended)": "openai/gpt-oss-120b",
    "GPT-OSS 20B (fastest)": "openai/gpt-oss-20b",
    "Qwen3.6 27B (strong reasoning)": "qwen/qwen3.6-27b",
    "Llama 4 Scout 17B": "meta-llama/llama-4-scout-17b-16e-instruct",
}

QUICK_QUESTIONS = [
    "Best universities in Germany for CS",
    "MS tuition fees in Canada",
    "F-1 student visa documents checklist",
    "Cost of living in the UK for students",
    "Scholarships for international students",
    "Schengen visa processing time",
]

SYSTEM_PROMPT = """You are StudyBuddy AI, an expert virtual counselor specializing in
international education and immigration. You help students with two broad categories
of questions:

1. STUDYING ABROAD
   - Universities/colleges by country, city, and program
   - Tuition fees for Bachelor's, Master's, and PhD programs
   - Typical degree duration
   - Living costs: accommodation, food, transport, and other monthly expenses
   - University locations (city/country)
   - Admission requirements and step-by-step application processes
   - Scholarships and financial aid opportunities

2. VISAS & IMMIGRATION
   - Student visa requirements and application steps
   - Visitor/tourist visa requirements and procedures
   - Visa application fees
   - Required documents
   - Processing times
   - Eligibility criteria
   - Other travel/immigration guidelines

RESPONSE STYLE:
- Be accurate, clear, and well-organized. Use headings, bullet points, and short
  paragraphs rather than dense blocks of text.
- Use tables when comparing multiple universities, countries, or visa types.
- When numeric figures (fees, costs, processing times) are involved, present your
  best current estimate but always note that these change frequently and the
  reader should confirm exact figures on the relevant university or government
  website before making decisions.
- If a question is ambiguous (e.g., no country or degree level specified), ask a
  brief clarifying question before giving a full answer, or state the assumption
  you're making and proceed.
- Never fabricate specific scholarship names, deadlines, or visa fee amounts you
  are not reasonably confident about — flag them as approximate instead.
- Keep tone friendly, encouraging, and professional, like a knowledgeable study-abroad
  counselor.
- Do not answer questions unrelated to studying abroad, university admissions,
  travel, or visas/immigration — politely redirect the user back to these topics.
"""

# ----------------------------------------------------------------------
# Theme: modern color system + typography
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --sb-primary: #6C5CE7;
        --sb-primary-dark: #5A4BD6;
        --sb-accent: #00CEC9;
        --sb-bg: #F5F6FB;
        --sb-bg-2: #ECEEFA;
        --sb-sidebar-bg: #14162B;
        --sb-sidebar-bg-2: #1C1E3A;
        --sb-sidebar-border: #2A2D52;
        --sb-text-muted: #A0A3C4;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}

    .stApp {
        background: linear-gradient(180deg, var(--sb-bg) 0%, var(--sb-bg-2) 100%);
    }

    /* ---------------- SIDEBAR ---------------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--sb-sidebar-bg) 0%, var(--sb-sidebar-bg-2) 100%);
        border-right: 1px solid var(--sb-sidebar-border);
    }
    section[data-testid="stSidebar"] * {
        color: #E7E8F5 !important;
        font-family: 'Inter', sans-serif;
    }
    section[data-testid="stSidebar"] input {
        color: #14162B !important;
    }
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        font-family: 'Poppins', sans-serif;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em;
        color: #C9CBEF !important;
        text-transform: uppercase;
        opacity: 0.85;
    }

    /* Sidebar logo */
    .sb-logo-wrap {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.4rem 0 1.1rem 0;
        margin-bottom: 0.6rem;
        border-bottom: 1px solid var(--sb-sidebar-border);
    }
    .sb-logo-badge {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: linear-gradient(135deg, var(--sb-primary) 0%, var(--sb-accent) 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        color: white !important;
        box-shadow: 0 4px 14px rgba(108, 92, 231, 0.45);
        flex-shrink: 0;
    }
    .sb-logo-text {
        line-height: 1.1;
    }
    .sb-logo-text .name {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 1.15rem;
        color: #FFFFFF !important;
    }
    .sb-logo-text .tagline {
        font-size: 0.72rem;
        color: var(--sb-text-muted) !important;
    }

    /* Uniform button styling in sidebar (chat history, quick questions, clear btn) */
    section[data-testid="stSidebar"] .stButton button {
        background-color: rgba(255,255,255,0.04) !important;
        color: #E7E8F5 !important;
        border: 1px solid var(--sb-sidebar-border) !important;
        border-radius: 10px !important;
        text-align: left !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 0.45rem 0.8rem !important;
        transition: all 0.15s ease-in-out;
        width: 100%;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background: linear-gradient(135deg, var(--sb-primary) 0%, var(--sb-accent) 100%) !important;
        border-color: transparent !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(108, 92, 231, 0.35);
    }
    section[data-testid="stSidebar"] .stButton button:focus:not(:hover) {
        background-color: rgba(255,255,255,0.04) !important;
        color: #E7E8F5 !important;
        border-color: var(--sb-sidebar-border) !important;
        box-shadow: none !important;
    }
    /* Highlight the active chat */
    .active-chat button {
        background: rgba(108, 92, 231, 0.22) !important;
        border-color: var(--sb-primary) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    /* New chat / primary action button */
    section[data-testid="stSidebar"] .sb-new-chat button {
        background: linear-gradient(135deg, var(--sb-primary) 0%, var(--sb-primary-dark) 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        text-align: center !important;
    }
    /* Clear conversation (danger) button */
    section[data-testid="stSidebar"] .sb-clear button {
        background-color: rgba(255, 90, 95, 0.08) !important;
        border: 1px solid rgba(255, 90, 95, 0.4) !important;
        color: #FF8A8E !important;
        text-align: center !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] .sb-clear button:hover {
        background: #FF5A5F !important;
        border-color: #FF5A5F !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(255, 90, 95, 0.4);
    }

    /* Sidebar footer */
    .sb-sidebar-footer {
        margin-top: 1.2rem;
        padding-top: 0.8rem;
        border-top: 1px solid var(--sb-sidebar-border);
        font-size: 0.72rem;
        color: var(--sb-text-muted) !important;
        line-height: 1.5;
    }

    /* ---------------- MAIN AREA ---------------- */
    .hero {
        padding: 1.3rem 1.7rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #6C5CE7 0%, #00B4D8 100%);
        color: white;
        margin-bottom: 1.3rem;
        box-shadow: 0 8px 24px rgba(108, 92, 231, 0.28);
    }
    .hero h1 {
        margin: 0 0 0.3rem 0;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 1.75rem;
    }
    .hero p {
        margin: 0;
        opacity: 0.94;
        font-size: 0.98rem;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 0.5rem 0.3rem;
    }
    .stChatMessage p, .stChatMessage li {
        font-size: 0.96rem;
        line-height: 1.55;
    }

    /* App footer */
    .app-footer {
        text-align: center;
        padding: 1.1rem 0 0.4rem 0;
        margin-top: 1.5rem;
        border-top: 1px solid rgba(108, 92, 231, 0.15);
        font-size: 0.78rem;
        color: #8A8DAE;
    }
    .app-footer b { color: #6C5CE7; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Session state: multi-chat history
# ----------------------------------------------------------------------
def _new_chat(select=True):
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {"title": "New chat", "messages": []}
    if select:
        st.session_state.current_chat_id = chat_id
    return chat_id


if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.current_chat_id = _new_chat()

if "api_key" not in st.session_state:
    st.session_state.api_key = os.environ.get("GROQ_API_KEY", "")

if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = None

current_chat = st.session_state.chats[st.session_state.current_chat_id]

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    # --- Logo ---
    st.markdown(
        """
        <div class="sb-logo-wrap">
            <div class="sb-logo-badge">SBai</div>
            <div class="sb-logo-text">
                <div class="name">StudyBuddy AI</div>
                <div class="tagline">Study Abroad & Visa Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- New chat ---
    st.markdown('<div class="sb-new-chat">', unsafe_allow_html=True)
    if st.button("➕ New chat", use_container_width=True, key="new_chat_btn"):
        _new_chat()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Chat history ---
    st.markdown("### 🕘 Previous chats")
    if len(st.session_state.chats) == 0:
        st.caption("No chats yet.")
    else:
        # newest first
        for chat_id in reversed(list(st.session_state.chats.keys())):
            chat = st.session_state.chats[chat_id]
            is_active = chat_id == st.session_state.current_chat_id
            wrapper_class = "active-chat" if is_active else ""
            st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
            label = ("📍 " if is_active else "💬 ") + chat["title"]
            if st.button(label, key=f"chat_{chat_id}", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # --- Quick / default questions ---
    st.markdown("### 💡 Quick questions")
    for i, q in enumerate(QUICK_QUESTIONS):
        if st.button(q, key=f"quick_{i}", use_container_width=True):
            st.session_state.queued_prompt = q

    st.markdown("---")

    # --- Settings ---
    with st.expander("Setting", expanded=False, icon="⚙"):
        api_key_input = st.text_input(
            "Groq API Key",
            value=st.session_state.api_key,
            type="password",
            help="Get a free key at console.groq.com/keys. Kept only in this session.",
        )
        st.session_state.api_key = api_key_input

        model_label = st.selectbox("Model", list(MODELS.keys()), index=0)
        model_id = MODELS[model_label]

        temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.4, 0.1)

    # --- Clear conversation ---
    st.markdown('<div class="sb-clear">', unsafe_allow_html=True)
    if st.button("🗑️ Clear conversation", use_container_width=True, key="clear_btn"):
        current_chat["messages"] = []
        current_chat["title"] = "New chat"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Sidebar footer ---
    st.markdown(
        """
        <div class="sb-sidebar-footer">
            StudyBuddy AI can make mistakes on fast-changing figures like fees and
            processing times. Always verify on official university/embassy sites.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎓 StudyBuddy AI</h1>
        <p>Your real-time assistant for universities, tuition, scholarships, and student/visitor visas.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Render existing chat history
# ----------------------------------------------------------------------
if not current_chat["messages"]:
    with st.chat_message("assistant", avatar="🎓"):
        st.markdown(
            "Hi! I'm **StudyBuddy AI** 👋. Ask me anything about studying abroad — "
            "universities, tuition, degree duration, living costs, admissions, "
            "scholarships — or about student and visitor visas. You can also pick "
            "a quick question from the sidebar. What are you planning?"
        )

for msg in current_chat["messages"]:
    avatar = "🎓" if msg["role"] == "assistant" else "🧑‍🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ----------------------------------------------------------------------
# Chat input & response generation
# ----------------------------------------------------------------------
typed_prompt = st.chat_input("Ask about universities, tuition, scholarships, or visas...")
prompt = typed_prompt or st.session_state.queued_prompt
st.session_state.queued_prompt = None  # consume it either way

if prompt:
    if not st.session_state.api_key:
        st.error("Please enter your Groq API key in the sidebar (Settings) to start chatting.")
        st.stop()

    current_chat["messages"].append({"role": "user", "content": prompt})
    if current_chat["title"] == "New chat":
        current_chat["title"] = (prompt[:38] + "…") if len(prompt) > 38 else prompt

    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    client = Groq(api_key=st.session_state.api_key)

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
        {"role": m["role"], "content": m["content"]} for m in current_chat["messages"]
    ]

    with st.chat_message("assistant", avatar="🎓"):
        placeholder = st.empty()
        full_response = ""
        try:
            stream = client.chat.completions.create(
                model=model_id,
                messages=api_messages,
                temperature=temperature,
                max_tokens=1800,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

        except RateLimitError:
            full_response = (
                "⏳ I've hit Groq's rate limit. Please wait a few seconds and try again."
            )
            placeholder.error(full_response)
        except APIConnectionError:
            full_response = (
                "🔌 I couldn't connect to Groq. Please check your internet connection "
                "and try again."
            )
            placeholder.error(full_response)
        except APIError as e:
            full_response = f"⚠️ Groq API error: {e}"
            placeholder.error(full_response)
        except Exception as e:
            full_response = f"⚠️ Something went wrong: {e}"
            placeholder.error(full_response)

    current_chat["messages"].append({"role": "assistant", "content": full_response})

# ----------------------------------------------------------------------
# App footer
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        © 2026 <b>StudyBuddy AI</b>. All rights reserved. &nbsp;|&nbsp; Developed by <b>Your Name</b>
    </div>
    """,
    unsafe_allow_html=True,
)