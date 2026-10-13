"""Streamlit web app: ask questions about HR policies and see the exact pages the answer came from.

Run:  streamlit run app.py
"""

import re
import sys
from pathlib import Path

import streamlit as st

# Lets the app run on hosts that install requirements.txt but not this package itself.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from policy_assistant.config import Settings
from policy_assistant.pipeline import PolicyAssistant

st.set_page_config(page_title="HR Policy Assistant", page_icon="📘", layout="centered")

EXAMPLES = [
    "How many sick days do I get, and when do I need a medical certificate?",
    "What is the hotel limit in Bengaluru?",
    "Can I work from outside India?",
    "What is my notice period after confirmation?",
    "How much leave can I carry forward?",
]
STATUS_LABEL = {
    "answered": ("Answered from the policies", "green"),
    "not_found": ("Not covered by the documents", "orange"),
    "unverified": ("Answer has no citation: check the sources", "red"),
}


@st.cache_resource(show_spinner="Indexing the sample policies...")
def sample_assistant() -> PolicyAssistant:
    return PolicyAssistant(Settings.from_env())


def current_assistant() -> PolicyAssistant:
    """Uploads stay private to the visitor's session; everyone else keeps the sample set."""
    return st.session_state.get("own_assistant") or sample_assistant()


def md_escape(text: str) -> str:
    """Show document text literally: stop "3. Hotels" becoming a list or "$" starting maths."""
    text = re.sub(r"([\\`*_{}\[\]<>#+|$~])", r"\\\1", text)
    return re.sub(r"^(\d+)\.", r"\1\\.", text)


def render_answer(result: dict) -> None:
    label, color = STATUS_LABEL.get(result["status"], (result["status"], "gray"))
    st.markdown(f":{color}-badge[{label}]")
    st.markdown(result["answer"])
    for c in result["citations"]:
        with st.expander(f"[{c['number']}] {c['doc_title']}, page {c['page']}"):
            st.caption(md_escape(c["snippet"]))
    tokens = result["prompt_tokens"] + result["completion_tokens"]
    st.caption(f"{result['latency_ms']} ms · {result['mode'] or 'no generation'}"
               + (f" · {tokens} tokens" if tokens else ""))


assistant = current_assistant()

with st.sidebar:
    st.header("Documents")
    for d in assistant.documents:
        st.markdown(f"**{d['title']}**  \n{d['pages']} pages · {d['chunks']} chunks")

    st.divider()
    uploads = st.file_uploader("Try your own PDFs", type="pdf", accept_multiple_files=True,
                               help="Uploaded files are only used in your browser session.")
    replace = st.checkbox("Search only my uploads", value=False)
    if st.button("Index uploads", disabled=not uploads, use_container_width=True):
        base = sample_assistant()
        chunks = [] if replace else list(base.retriever.chunks)
        own = PolicyAssistant(base.settings, chunks=chunks, llm=base.llm)
        with st.spinner("Reading PDFs..."):
            for f in uploads:
                try:
                    own.add_pdf(f.getvalue(), f.name)
                except Exception as exc:
                    st.error(f"{f.name}: {exc}")
        st.session_state["own_assistant"] = own
        st.session_state["history"] = []
        st.rerun()
    if "own_assistant" in st.session_state and st.button("Back to sample policies",
                                                         use_container_width=True):
        del st.session_state["own_assistant"]
        st.session_state["history"] = []
        st.rerun()

    st.divider()
    llm = assistant.settings.llm_model if assistant.llm else "none: extractive quotes"
    st.caption(f"Search: {assistant.retriever.mode}  \nLLM: {llm}")
    st.caption("Sample documents belong to Brightwave Analytics, a fictional company.")

st.title("HR Policy Assistant")
st.write("Ask about leave, remote work, travel, conduct or notice periods. "
         "Every answer cites the document and page it came from, and the assistant "
         "says so when the policies don't cover a question.")

history = st.session_state.setdefault("history", [])
picked = st.pills("Try an example", EXAMPLES, selection_mode="single", key="example")
question = st.chat_input("Ask a policy question")
if picked and picked != st.session_state.get("last_example"):
    st.session_state["last_example"] = picked
    question = question or picked

for item in history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        render_answer(item)

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the policies..."):
            result = assistant.ask(question).to_dict()
        render_answer(result)
    history.append(result)
