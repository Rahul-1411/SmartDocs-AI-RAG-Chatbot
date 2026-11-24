

import os
import time

import streamlit as st

from rag import SUPPORTED, RAG

st.set_page_config(page_title="Document Chat", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading index...")
def get_rag() -> RAG:
    return RAG()


rag = get_rag()

ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("uploader_n", 0)
ss.setdefault("confirm_clear", False)
ss.setdefault("flash", None)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
    )

    st.header("Add documents")
    files = st.file_uploader(
        "PDF, DOCX, TXT, MD, CSV",
        type=SUPPORTED,
        accept_multiple_files=True,
        key=f"uploader_{ss.uploader_n}",
    )
    if st.button("Add to knowledge base", disabled=not files, use_container_width=True):
        notes = []
        total_files = len(files)
        bar = st.progress(0.0, text="Preparing ingestion...")
        progress_detail = st.empty()
        started = time.monotonic()
        for i, f in enumerate(files):
            def report_progress(stage, current, total):
                ratio = current / total if total else 0.0
                if stage in ("Reading document", "Reading PDF pages"):
                    file_progress = 0.2 * ratio
                elif stage == "Splitting into chunks":
                    file_progress = 0.2 + 0.1 * ratio
                elif stage == "Embedding chunks":
                    file_progress = 0.3 + 0.65 * ratio
                else:
                    file_progress = 0.95 + 0.05 * ratio

                overall_progress = (i + file_progress) / total_files
                elapsed = time.monotonic() - started
                eta = ""
                if overall_progress >= 0.05:
                    remaining = int(elapsed * (1 - overall_progress) / overall_progress)
                    minutes, seconds = divmod(remaining, 60)
                    eta = f" · about {minutes}m {seconds:02d}s remaining"

                bar.progress(
                    min(overall_progress, 1.0),
                    text=f"File {i + 1}/{total_files}: {f.name} · {stage}{eta}",
                )
                if stage in ("Embedding chunks", "Reading PDF pages"):
                    progress_detail.caption(
                        f"{stage}: {current:,}/{total:,} processed · "
                        f"{max(total - current, 0):,} remaining"
                    )
                elif stage == "Splitting into chunks":
                    progress_detail.caption(
                        f"Preparing section {current:,}/{total:,}"
                    )

            try:
                res = rag.add_document(f.name, f.getvalue(), report_progress)
                if res["status"] == "exists":
                    notes.append(f"⚠️ {f.name}: already added")
                else:
                    notes.append(f"✅ {f.name}: {res['chunks']} chunks")
            except Exception as e:
                notes.append(f"❌ {f.name}: {e}")
            bar.progress((i + 1) / total_files, text=f"Finished {i + 1}/{total_files} files")
        ss.flash = notes
        ss.uploader_n += 1  # resets the uploader
        st.rerun()

    if ss.flash:
        for n in ss.flash:
            st.write(n)
        ss.flash = None

    st.header(f"Documents ({len(rag.docs)})")
    if not rag.docs:
        st.caption("Nothing uploaded yet.")
    for doc_id, d in list(rag.docs.items()):
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{d['name']}**  \n{d['chunks']} chunks · {d['added']}")
        if c2.button("🗑", key=f"del_{doc_id}", help="Delete this document"):
            rag.delete_document(doc_id)
            st.rerun()

    if rag.docs:
        if not ss.confirm_clear:
            if st.button("Delete all documents", use_container_width=True):
                ss.confirm_clear = True
                st.rerun()
        else:
            st.warning("Delete every document and its index?")
            y, n = st.columns(2)
            if y.button("Yes, delete", type="primary"):
                rag.clear()
                ss.messages = []
                ss.confirm_clear = False
                st.rerun()
            if n.button("Cancel"):
                ss.confirm_clear = False
                st.rerun()

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        ss.messages = []
        st.rerun()

# ---------------------------------------------------------------- chat
st.title("📚 Document Chat")
st.caption("Ask questions across everything in your knowledge base.")


def show_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for n, s in enumerate(sources, 1):
            page = f", p.{s['page']}" if s["page"] else ""
            st.markdown(f"**[{n}] {s['name']}{page}**")
            st.caption(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))


for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        show_sources(m.get("sources"))

if not rag.docs:
    st.info("Upload documents in the sidebar to get started.")

question = st.chat_input(
    "Ask a question about your documents...",
    disabled=not rag.docs or not api_key.strip(),
)
if rag.docs and not api_key.strip():
    st.warning("Add your Gemini API key in the sidebar (or set GEMINI_API_KEY in .env).")

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in ss.messages]
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents..."):
                query = rag.condense(question, history, api_key)
                hits = rag.retrieve(query)
            answer = st.write_stream(rag.stream_answer(question, hits, history, api_key))
            sources = [{"name": h.doc_name, "page": h.page, "text": h.text} for h in hits]
            show_sources(sources)
        except Exception as e:
            answer, sources = f"Error: {e}", []
            st.error(answer)
    ss.messages.append({"role": "assistant", "content": answer, "sources": sources})
