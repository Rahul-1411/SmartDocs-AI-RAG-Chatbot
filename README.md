# Document Chat — Retrieval-Augmented Generation (RAG)

A local document-question-answering app built with **Streamlit**, **Sentence Transformers**, **NumPy**, and the **Google Gemini API**.

Upload supported documents, let the app extract and index their text, then ask questions across the uploaded knowledge base. Answers are generated with Gemini using the passages retrieved from the local index, and the UI displays the source document and page (when available).

> **Scope note:** This README describes the implementation in `app.py` and `rag.py`. It does not claim features that are not present in the code, such as user accounts, a REST API, OCR, cloud vector databases, hybrid search, or multi-user isolation.

## Features

- **Multiple document formats:** upload PDF, DOCX, TXT, Markdown (`.md`), and CSV files.
- **Batch upload:** add multiple files in one ingestion run.
- **Text extraction:**
  - PDF: extracts selectable text page by page and keeps page numbers.
  - DOCX: extracts paragraphs and table rows.
  - CSV: reads the first row as headers and turns each subsequent row into labelled text.
  - TXT and Markdown: decode the file contents as text.
- **Chunking with overlap:** splits extracted text into approximately 180-word chunks with a 35-word overlap to retain nearby context.
- **Local semantic embeddings:** uses `BAAI/bge-small-en-v1.5` through Sentence Transformers. Embeddings are normalized and generated in batches of 32.
- **Similarity-based retrieval:** embeds the question and ranks stored chunks using cosine similarity (implemented as a dot product over normalized vectors). The top 6 chunks are selected.
- **Follow-up question handling:** optionally asks Gemini to rewrite a follow-up into a standalone search query using recent chat context.
- **Grounded answer generation:** sends retrieved passages to Gemini and instructs it to answer only from those passages, state when the documents do not contain the answer, and cite excerpts inline as `[1]`, `[2]`.
- **Streaming responses:** displays generated answer text as it arrives from Gemini.
- **Source inspection:** shows retrieved source excerpts and their document names; PDF page numbers are shown when available.
- **Persistent local index:** saves document metadata/chunks in `data/store.json` and vectors in `data/embeddings.npy`, so the indexed knowledge base can be loaded again after restarting the app.
- **Document management:** view indexed documents, delete one document, or clear the full knowledge base.
- **Ingestion progress:** shows per-file stages and progress while documents are processed.

## How it works

The project has two main files:

- `app.py` — Streamlit user interface, upload flow, chat history, progress display, and document management.
- `rag.py` — document extraction, chunking, embedding, local persistence, retrieval, query condensation, and Gemini response generation.

### Document ingestion pipeline

```text
User selects one or more files in Streamlit
                 |
                 v
       Check file extension
                 |
                 v
     Extract text by file type
     PDF   -> text per page
     DOCX  -> paragraphs + tables
     CSV   -> header-labelled rows
     TXT/MD-> decoded text
                 |
                 v
       Split text into chunks
       ~180 words, 35 overlap
                 |
                 v
  Prefix each chunk with its filename
                 |
                 v
 Generate normalized BGE-small embeddings
       Sentence Transformers, batches of 32
                 |
                 v
 Append chunks + vectors to local index
                 |
                 v
 Persist metadata/chunks and vectors
 data/store.json + data/embeddings.npy
```

A file is identified using the first 12 characters of the SHA-1 hash of its contents. If the same file contents are already indexed, the app reports that the file already exists instead of indexing it again. This means identical content under a different filename is also treated as already added.

### Question-answering pipeline

```text
User asks a question
        |
        v
Read recent conversation history
        |
        v
If there is history:
ask Gemini to rewrite follow-up as
a standalone retrieval query
        |
        v
Embed the retrieval query locally
        |
        v
Compare query vector with stored vectors
(dot product; vectors are normalized)
        |
        v
Select the top 6 matching chunks
        |
        v
Build context with source labels
(document name and page when available)
        |
        v
Send question + context + recent chat
to Gemini using the configured model
        |
        v
Stream answer into the Streamlit chat
        |
        v
Show retrieved source excerpts in the UI
```

The rewritten query is used for retrieval. The original user question is sent to Gemini for answer generation along with the retrieved context and recent conversation history.

## Technology used

| Area | Implementation |
|---|---|
| UI | Streamlit |
| Language | Python |
| Document parsing | `pypdf`, `python-docx`, Python CSV module |
| Embeddings | Sentence Transformers |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Vector operations | NumPy |
| LLM | Google Gemini via `google-genai` |
| Default generation model | `gemini-2.5-flash` |
| Configuration | `python-dotenv` |
| Local persistence | JSON metadata/chunks + NumPy `.npy` vectors |

The embedding model runs locally through Sentence Transformers. Gemini is used for query condensation (when chat history exists) and answer generation, so a Gemini API key and internet access are required for chat responses.

## Requirements

- Python **3.10 or newer** is a practical starting point for the listed dependencies.
- `pip`
- Internet access for installing dependencies, downloading the embedding model on its first use, and calling Gemini.
- A Google Gemini API key.

The repository does not include a pinned Python version or a lockfile. Dependency versions are specified in `requirements.txt` using minimum versions.

## Installation and setup

### 1. Get the project files

Clone the repository if it is hosted in Git, or extract the project ZIP. Open a terminal in the directory containing `app.py`, `rag.py`, and `requirements.txt`.

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a file named `.env` in the same directory as `app.py` and `rag.py`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

You can obtain an API key from [Google AI Studio](https://aistudio.google.com/).

The app also provides a password-style Gemini API key field in the Streamlit sidebar. Alternatively, the key can be supplied through the `.env` file. The code reads `LLM_MODEL` from the environment and defaults to `gemini-2.5-flash`, so you can optionally override it:

```env
GEMINI_API_KEY=your_gemini_api_key_here
LLM_MODEL=gemini-2.5-flash
```

Do not commit `.env` or share your API key. The project `.gitignore` already excludes `.env`.

### 5. Run the app

From the project directory:

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal (usually `http://localhost:8501`). Open that URL in your browser.

### 6. Use the app

1. Enter your Gemini API key in the sidebar if it is not already loaded from `.env`.
2. Select one or more PDF, DOCX, TXT, MD, or CSV files.
3. Click **Add to knowledge base** and wait for ingestion to finish.
4. Ask a question in the chat input.
5. Expand **Sources** below a response to inspect the retrieved passages.
6. Use the sidebar to delete individual documents, clear all indexed documents, or clear the current chat.

On the first ingestion, Sentence Transformers may download the embedding model. This can take time and requires internet access. The app loads the saved local index on subsequent starts.

## Configuration

The current configuration is defined in `rag.py`:

| Setting | Current value | Meaning |
|---|---:|---|
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model |
| `LLM_MODEL` | `gemini-2.5-flash` by default | Gemini model used for condensation and generation |
| `CHUNK_WORDS` | `180` | Approximate target words per chunk |
| `OVERLAP_WORDS` | `35` | Approximate overlap carried between chunks |
| `TOP_K` | `6` | Number of chunks retrieved per question |
| `HISTORY_TURNS` | `6` | Maximum recent exchanges passed to generation |

These values can be changed in `rag.py`. Changing the embedding model makes the existing index incompatible with the loader's model check; the app will not load vectors created with a different model. Re-index documents after changing the embedding model.

## Project structure

```text
project/
├── app.py                 # Streamlit interface and chat flow
├── rag.py                 # RAG pipeline and local index implementation
├── requirements.txt       # Python dependencies
├── .env                   # Local secrets; create this yourself
├── .gitignore
└── data/
    ├── store.json         # Indexed document metadata and text chunks
    └── embeddings.npy     # Corresponding embedding vectors
```

The `data/` files are created/updated by the application. They are local runtime data, not source code. The supplied project archive includes data files, but for a clean setup you can let the app create its own index by uploading documents.

## Important implementation details and limitations

- **No OCR:** PDF processing uses `pypdf` text extraction. Scanned/image-only PDFs may produce no readable text and cannot be indexed by this implementation.
- **No Excel support:** supported extensions are PDF, DOCX, TXT, MD, and CSV. `.xlsx` and `.xls` are not supported.
- **CSV is text indexed:** CSV rows are converted into labelled text. The project does not create a SQL database or perform structured SQL queries over CSV data.
- **No hybrid or keyword search:** retrieval is semantic vector similarity only; there is no BM25 or keyword-ranking stage.
- **No external vector database:** vectors are stored in a local NumPy file and searched in-process.
- **No API/backend service:** the app is a Streamlit application; the code does not expose a REST API.
- **No authentication or multi-user separation:** the implementation does not include user accounts, authorization, or tenant-isolated indexes.
- **Local persistence is not a backup system:** deleting the `data/` directory removes the saved index. Keep original documents if you may need to rebuild it.
- **Source display reflects retrieved chunks:** the UI displays the passages selected for the question. This is useful for checking grounding, but it does not guarantee that every generated claim is correct.
- **Text extraction quality matters:** complex layouts, images, diagrams, handwriting, and unusual document formatting may not be represented well in extracted text.
- **Chat history is Streamlit session state:** the indexed knowledge base is persisted to disk, but chat history is not written to the same persistent index files.

## Troubleshooting

**`Missing Gemini API key`**
- Add `GEMINI_API_KEY` to `.env`, or enter the key in the sidebar.
- Restart the app after changing environment configuration if needed.

**A PDF uploads but no text is indexed**
- It may be a scanned/image-only PDF. This project does not currently run OCR.
- Try a text-selectable PDF.

**The first upload is slow**
- The embedding model may be downloading, and embedding is performed locally. Later runs can reuse the downloaded model cache.

**A file says it was already added**
- The app uses a hash of file contents to detect duplicates. Identical contents are treated as the same document.

**The knowledge base appears empty after changing the embedding model**
- The loader checks that the saved index model matches the configured embedding model. Rebuild the index with the current model.

**Gemini requests fail**
- Check that the API key is valid, that the selected model is available to your key, and that the machine has internet access.

## Security and data handling

- Keep the Gemini API key private and out of source control.
- Document text and retrieved context are sent to Gemini when generating answers. Follow your organization's data-handling requirements before uploading confidential documents.
- The local index contains extracted document text and embeddings. Protect the `data/` directory accordingly.
- This project does not implement user authentication or access controls.

## License

No license file was present in the supplied project files. Add a license only after deciding how you want others to use, modify, and distribute the project.
