---
title: ChatUGM
emoji: 🎓
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: University of Greater Manchester AI assistant with Study Mode
---

# 🎓 ChatUGM

> **The University of Greater Manchester's AI assistant. Answers questions about the university, and doubles as a document-aware study companion for students.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-FF7C00)](https://gradio.app/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/transformers)
[![Llama 3.2](https://img.shields.io/badge/LLM-Llama%203.2%201B-8B5CF6)](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Deployed](https://img.shields.io/badge/🤗-HF%20Spaces-yellow)](https://huggingface.co/spaces)

---

## Table of Contents

- [Overview](#overview)
- [Meet Axiom](#meet-axiom)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Study Mode Deep Dive](#study-mode-deep-dive)
- [Supported File Formats](#supported-file-formats)
- [Intents Database](#intents-database)
- [Chat Logging](#chat-logging)
- [UI and Theming](#ui-and-theming)
- [Configuration](#configuration)
- [Installation](#installation)
- [Running Locally](#running-locally)
- [Deployment on Hugging Face Spaces](#deployment-on-hugging-face-spaces)
- [How to Use ChatUGM](#how-to-use-chatugm)
- [Design Decisions](#design-decisions)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Roadmap](#roadmap)
- [Author](#author)
- [License](#license)

---

## Overview

**ChatUGM** is a Gradio-based AI assistant built for the University of Greater Manchester. It handles two very different jobs in one clean interface:

1. **University Q&A** — Answers questions about courses, fees, admissions, campus facilities, accommodation, student services, and everything else a student or prospective applicant might ask about UGM. Built on a curated intent database of 40+ categories scraped from official university sources.

2. **Study Mode** — A document-aware study companion. Students upload their lecture notes, assignments, readings, datasets, or textbook chapters, and ChatUGM becomes a personal tutor that answers questions from the uploaded material, breaks down assignments step-by-step, or generates comprehensive summaries.

The chatbot's persona is **Axiom**, an AI assistant branded specifically for UGM. Behind the friendly name is a hybrid retrieval-and-generation architecture that combines fast TF-IDF intent matching with a Llama 3.2 1B Instruct fallback for anything the intent database doesn't cover.

---

## Meet Axiom

Axiom is the chatbot's persona. When users ask "who built you?" or "who are you?", Axiom identifies itself as ChatUGM's AI assistant. The system prompt sets Axiom's character:

> *"You are Axiom, The University of Greater Manchester's helpful assistant."*

Axiom is designed to be friendly, concise, helpful, and honest when it doesn't know something. The tone is conversational, and emojis are used sparingly to keep interactions warm without feeling childish.

---

## Key Features

| Capability | Details |
|------------|---------|
| **Hybrid Architecture** | Fast TF-IDF intent matching for curated FAQ content, Llama 3.2 1B Instruct fallback for open questions |
| **40+ Intent Categories** | Courses, fees, admissions, accommodation, student services, campus info, contact details, and more |
| **Study Mode** | Document-aware Q&A with three modes: Interact, Assignment, Summary |
| **Multi-format Document Support** | PDF, Word (.docx), PowerPoint (.pptx), Excel (.xlsx), CSV, JSON, TXT |
| **Intelligent Chunking** | Documents split into 100–500 character semantic chunks with TF-IDF retrieval |
| **Assignment Breakdown** | Automated assignment analysis with task extraction, requirement parsing, deadline detection, and suggested approach |
| **Comprehensive Summaries** | LLM-generated summaries with extractive fallback that pulls key sentences, concepts, and statistics |
| **Chat Logging** | Every interaction logged to CSV for analytics and improvement |
| **Session Management** | Each chat session gets a unique UUID for traceable conversation history |
| **UGM Branded UI** | University colour gradient header (blue to red), dark theme, graduation cap reset button |
| **Mobile Responsive** | Fully responsive layout tested on phones, tablets, and desktops |
| **Fixed Bottom Input** | Chat input stays pinned to the bottom of the viewport like modern messaging apps |
| **Study Mode Toggle** | One-click toggle between normal chat and document-aware study mode |
| **CPU Fallback** | Runs on free-tier HF Spaces CPU when GPU is unavailable |

---

## Tech Stack

```
Frontend & UI:           Gradio 4.x with custom CSS
LLM:                     Llama 3.2 1B Instruct (meta-llama/Llama-3.2-1B-Instruct)
Intent Matching:         scikit-learn TF-IDF + cosine similarity
Model Runtime:           Hugging Face Transformers + PyTorch
Document Processing:     pdfplumber, python-docx, python-pptx, pandas
Data Layer:              Curated intents.json + CSV logging
Deployment:              Hugging Face Spaces (Gradio SDK)
```

No external APIs, no vector database, no cloud services beyond Hugging Face. Everything runs inside the Space.

---

## Architecture

ChatUGM uses a **two-stage response pipeline** that balances speed, accuracy, and cost.

### Stage 1: Intent Matching (TF-IDF + Cosine Similarity)

When a user sends a message, the system first checks it against the intent database (`intents.json`), which contains 40+ categories of curated UGM-specific Q&A. Each intent has a `tag`, a set of `patterns` (example phrasings), and a set of `responses`.

The matching process:

1. User text is preprocessed (lowercased, whitespace collapsed)
2. A TF-IDF vectoriser (fitted on all patterns with 1–2 gram range) transforms the text
3. Cosine similarity is computed against every pattern in the corpus
4. If the highest similarity score exceeds **0.45** (the threshold), that intent's response is returned immediately

This is fast (microseconds), deterministic, and perfect for questions like "what courses do you offer" or "how much are the fees", where the answer is curated and authoritative.

### Stage 2: LLM Fallback (Llama 3.2 1B Instruct)

If no intent matches above the threshold, the question is routed to Llama 3.2 1B Instruct. The model is loaded via Hugging Face Transformers with:

- `device_map="auto"` for GPU utilisation when available
- `torch.float16` precision to cut memory
- CPU fallback on free-tier Spaces

The last 6 messages of conversation history are included in the prompt to preserve context, and the model generates up to 200 new tokens with temperature 0.7 and top-p 0.9.

### Why This Hybrid Approach

A pure LLM approach would burn GPU time on questions already answered in the intent database, and would occasionally hallucinate specific facts (fees, deadlines, contact details). A pure intent database would fail on novel questions the curator didn't anticipate. The hybrid approach gets the best of both: authoritative answers on known topics, flexible LLM answers on unknown ones, and a clean audit trail (every interaction logs whether it was matched by an intent or answered by the LLM).

---

## Study Mode Deep Dive

Study Mode is the feature that turns ChatUGM from a university FAQ bot into a full study companion. Toggle it on with the 📚 button in the top right, upload a document with the 📎 button, and the bot becomes your personal tutor for whatever you just uploaded.

### Activation Flow

1. User clicks the 📚 toggle button. A message confirms Study Mode is on
2. The 📎 file upload button becomes visible
3. User uploads a document. The system extracts text, chunks it, and confirms the document is loaded
4. User picks a mode by typing one of three commands

### The Three Modes

**💬 Interact Mode** (type `interact`)

Free-form Q&A against the uploaded document. The user asks questions in natural language, and ChatUGM retrieves the most relevant chunks via TF-IDF similarity, then either generates an LLM answer grounded in those chunks or falls back to presenting the most relevant passage directly.

Typical use: "What does the document say about neural networks?" or "Explain the third section in simpler terms."

**📝 Assignment Mode** (type `assignment`)

When a student uploads an assignment brief, this mode automatically breaks it down into:

- **Assignment Overview** (2–3 sentences)
- **Main Tasks** (extracted via regex patterns that detect numbered questions, Part 1/2/3 markers, lettered sub-tasks)
- **Requirements** (sentences containing "must", "should", "required", "need to", etc.)
- **Important Dates** (deadlines extracted via regex)
- **Key Topics** (named entities extracted from the document)
- **Suggested Approach** (7-step generic framework for tackling assignments)

After the breakdown, the student can ask follow-up questions. The system is explicitly instructed via prompt to **guide the student, not do the work for them** — helping them understand requirements and think through problems rather than just generating answers. This is the right pedagogical default for an AI study assistant.

**📄 Summary Mode** (type `summary`)

Generates a comprehensive document summary with:

- **Overview** (2–3 sentence introduction)
- **Key Points** (sentences containing importance markers or statistics)
- **Key Terms & Concepts** (named entities)
- **Key Figures** (numbers, percentages, dollar amounts)
- **Document Statistics** (word count, section count)
- **Main Takeaway** (the last meaningful sentence or explicit conclusion)

If the LLM is available, it generates the summary. Otherwise an extractive fallback produces a structured summary using pattern matching and rules.

### Exiting a Mode

Typing `exit`, `quit`, `stop`, `end`, or `done` leaves the current sub-mode and returns the user to the Study Mode options menu.

### Chunking and Retrieval

Uploaded documents are split into **100–500 character semantic chunks**, with fallback to sentence-level splitting if the document lacks natural paragraph breaks. When a user asks a question, the system uses a second TF-IDF vectoriser (with English stopwords removed) to find the top 3 most relevant chunks, which then become the context window for the LLM.

This is essentially a lightweight RAG (Retrieval-Augmented Generation) system without the vector database overhead. For short documents (a single lecture PDF, an assignment brief, a book chapter), it performs well and stays fast on CPU.

---

## Supported File Formats

Study Mode accepts the following document types:

| Format | Extension | Library Used | Notes |
|--------|-----------|--------------|-------|
| **PDF** | `.pdf` | `pdfplumber` | Best for lecture slides, academic papers, assignment briefs |
| **Word** | `.docx`, `.doc` | `python-docx` | Extracts paragraphs + table content |
| **PowerPoint** | `.pptx`, `.ppt` | `python-pptx` | Extracts text per slide with slide numbers |
| **Excel** | `.xlsx`, `.xls` | `pandas` + `openpyxl` | Multi-sheet support, preserves headers |
| **CSV** | `.csv` | `pandas` | For datasets, reading lists, structured content |
| **JSON** | `.json` | Built-in | Recursively flattens nested structures |
| **Text** | `.txt` | Built-in | Plain text files |

If a required library is missing at runtime, the system returns a clear error message instead of crashing.

---

## Intents Database

The `intents.json` file is the knowledge base for Stage 1 (TF-IDF matching). Each intent is a JSON object with this structure:

```json
{
  "tag": "courses",
  "patterns": [
    "what courses do you offer",
    "list of courses",
    "programmes available",
    "degrees you offer",
    "undergraduate courses",
    "postgraduate programmes"
  ],
  "responses": [
    "UGM offers a wide range of undergraduate and postgraduate programmes..."
  ]
}
```

Categories covered include (non-exhaustive):

- **Admissions:** entry requirements, how to apply, application deadlines
- **Courses:** undergraduate, postgraduate, foundation, part-time
- **Fees:** UK tuition, international fees, scholarships, payment plans
- **Accommodation:** halls, private housing, costs, application process
- **Campus:** locations, facilities, library hours, opening times
- **Student Services:** support, wellbeing, disability services, careers
- **International Students:** visa, English requirements, orientation
- **Research:** postgraduate research, funding, supervisors
- **Contact:** phone, email, addresses of departments
- **About:** history, ranking, values, leadership
- **Practical:** parking, transport, food, events, clubs

To add a new category, add a new intent object to `intents.json` and restart the app. The TF-IDF vectoriser rebuilds on startup.

---

## Chat Logging

Every interaction is logged to CSV files in the `chat_logs/` directory.

### `chat_history.csv` (Master Log)

| Column | Description |
|--------|-------------|
| `timestamp` | UTC ISO timestamp |
| `session_id` | UUID unique to each chat session |
| `user_text` | What the user typed |
| `assistant_text` | What the bot replied |
| `intent_tag` | Matched intent tag, or empty if LLM-generated |
| `score` | TF-IDF similarity score for the match |

### `questions.csv` (Question-Only Log)

Lighter log focused on user questions only. Useful for analysing what students ask most, and for improving the intents database over time.

The logs are designed to be compatible with pandas, Excel, and any analytics tool that reads CSV. No PII is required — the session UUID is the only identifier.

---

## UI and Theming

ChatUGM's visual design is built around the University of Greater Manchester's brand colours: a blue-to-red gradient header, dark content area, and careful use of white space.

### Visual Elements

- **Header:** Fixed blue-to-red gradient bar with the 🎓 reset button on the left and "ChatUGM" title centred
- **Study Controls:** Top-right corner, 📚 toggle button and 📎 file upload button, both compact and circular
- **Chat Area:** Dark `#0f0f10` background with blue user bubbles (`#003d80`) and dark grey assistant bubbles (`#1e1e22`)
- **Input Bar:** Fixed at the bottom of the viewport, rounded 24px corners, auto-expanding textarea with a circular blue send button (`#0058b0`)
- **Typography:** System UI stack with clean message spacing

### Responsive Breakpoints

Custom breakpoints are defined at:

- `max-width: 768px` (tablets)
- `max-width: 480px` (phones)
- `max-width: 400px` (small Android devices)
- `max-height: 500px` (landscape phones)

Each breakpoint adjusts button sizes, padding, and the header height so the interface stays usable on any device.

### Auto-scroll Behaviour

A JavaScript snippet in the Gradio HTML block keeps the chat scrolled to the bottom as new messages arrive, using `MutationObserver` to detect DOM changes and a 300ms interval fallback. This mirrors the behaviour users expect from WhatsApp, iMessage, and ChatGPT.

---

## Configuration

### Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `HF_TOKEN` | Hugging Face access token for gated models | Required for Llama 3.2 |

On Hugging Face Spaces, set this in **Settings → Variables and secrets**. Locally, export it in your shell:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Runtime Flags in `app.py`

- `MODEL_ID` — The LLM to use (default: `meta-llama/Llama-3.2-1B-Instruct`)
- `DEVICE_MAP` — GPU allocation strategy (default: `"auto"`)
- `TORCH_DTYPE` — Precision (default: `torch.float16`)
- `INTENTS_PATH` — Path to the intents JSON file (default: `"intents.json"`)
- `LOG_DIR` — Where to write chat logs (default: `"chat_logs"`)

---

## Installation

### Prerequisites

- Python 3.10 or higher
- A Hugging Face account with access to Llama 3.2 (approved via the model's page)
- Approximately 3 GB free disk space for dependencies and model weights

### Quick Install

```bash
# Clone the repository
git clone https://github.com/[your-username]/chatugm.git
cd chatugm

# Install dependencies
pip install -r requirements.txt

# Set your HF token
export HF_TOKEN=hf_your_token_here
```

---

## Running Locally

```bash
python app.py
```

The app launches on `http://127.0.0.1:7860`. The first run downloads Llama 3.2 1B weights (~2 GB), which takes a few minutes on a typical connection.

To expose it on your local network:

```python
# At the bottom of app.py
demo.launch(server_name="0.0.0.0", server_port=7860)
```

---

## Deployment on Hugging Face Spaces

ChatUGM deploys cleanly to Hugging Face Spaces using the Gradio SDK.

**Steps:**

1. Create a new Space, selecting **Gradio** as the SDK
2. Clone the Space repo locally
3. Copy `app.py`, `intents.json`, and `requirements.txt` into the repo root
4. Keep the YAML frontmatter at the top of `README.md` intact
5. Go to **Settings → Variables and secrets** and add `HF_TOKEN` as a secret
6. Push to the Space's remote — HF will build and deploy automatically

**Hardware notes:**

- **CPU Basic (free):** Works, but LLM responses are slow (~20–40 seconds per response). Intent-matched responses remain instant
- **CPU Upgrade:** Noticeably faster for LLM responses
- **T4 GPU:** Recommended for production. LLM responses in 2–5 seconds
- **A10G / L4:** Overkill for a 1B model but fastest option

Build time is typically 5–10 minutes due to PyTorch, Transformers, and document-processing libraries.

---

## How to Use ChatUGM

### Asking University Questions

Just type naturally. Examples:

- *"What courses do you offer?"*
- *"How do I apply for postgraduate study?"*
- *"What are the fees for international students?"*
- *"Where is the library?"*
- *"How do I contact student services?"*

If the question matches a curated intent, you get an instant authoritative answer. If not, Axiom generates a response using the LLM.

### Using Study Mode

**1.** Click the 📚 button in the top-right corner to turn Study Mode on.

**2.** Click the 📎 button and upload a document (PDF, Word, PowerPoint, Excel, CSV, JSON, or TXT).

**3.** Wait for the confirmation message. ChatUGM tells you how many sections it found.

**4.** Pick a mode by typing:

- `interact` — to ask questions about the document
- `assignment` — to break down an assignment brief step-by-step
- `summary` — to get a comprehensive summary

**5.** Ask anything. The bot stays in the mode you chose until you type `exit`.

**6.** Type a different mode command at any time to switch.

### Resetting the Chat

Click the 🎓 graduation cap button at the top-left to clear chat history, reset Study Mode, and start a fresh session.

---

## Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| **Matching threshold** | 0.45 cosine similarity | Low enough to catch paraphrased questions, high enough to avoid false positives like matching "fees" to a question about "feedback" |
| **LLM model size** | Llama 3.2 1B Instruct | Small enough to run on free HF Spaces CPU tier, large enough for coherent responses. 8B and 70B alternatives would price out students testing the app |
| **Hybrid architecture** | Intent matching first, LLM fallback | Authoritative answers on known topics, flexible answers on novel ones, clear audit trail in the logs |
| **Chunk size** | 100–500 characters | Balanced for TF-IDF retrieval. Smaller chunks lose context, larger chunks dilute retrieval precision |
| **Assignment mode prompt** | "Don't do the work for them" | The right pedagogical default. An AI that does homework for students isn't a tutor — it's a crutch |
| **Conversation history in LLM prompt** | Last 6 messages only | Preserves short-term context without hitting token limits on a 1B model |
| **CSV logging** | Every interaction, with session UUID | Enables offline analytics and intent-database improvement without requiring a database |
| **Fixed bottom input** | Always pinned to viewport bottom | Matches the UX pattern of every modern messaging app |
| **Mobile breakpoints at 768/480/400px** | Tailored for tablet, phone, small Android | Students access this on phones more than laptops — mobile had to be first-class |
| **Dark theme only** | No light mode | Students often study at night. Dark mode is the default-correct choice, and maintaining two themes doubles the CSS surface area |
| **System prompt persona** | "You are Axiom" | Gives the bot a consistent identity that users can reference ("can you help me, Axiom?") |

---

## Repository Structure

```
.
├── README.md                     # This file (with HF Spaces YAML frontmatter)
├── app.py                        # Main ChatUGM app (Gradio UI + bot logic + Study Mode)
├── intents.json                  # Curated intent database (40+ categories)
├── requirements.txt              # Python dependencies
├── chat_logs/                    # (generated) CSV logs
│   ├── chat_history.csv          # Master interaction log
│   └── questions.csv             # Questions-only log
└── LICENSE                       # MIT License
```

Single-file app by design. Everything needed to run, customise, or extend ChatUGM lives in `app.py`, with the knowledge base cleanly separated in `intents.json`.

---

## Dependencies

```
gradio>=4.44.0
transformers>=4.40.0
torch>=2.0.0
huggingface-hub>=0.20.0
scikit-learn>=1.4.0
pandas>=2.0.0
pdfplumber>=0.10.0
python-docx>=1.1.0
python-pptx>=0.6.23
openpyxl>=3.1.0
```

Install everything with:

```bash
pip install -r requirements.txt
```

---

## Roadmap

Features that may land in future versions:

- **Voice input and voice output** — Whisper for STT, Coqui or ElevenLabs for TTS
- **Multi-language support** — German, Arabic, Mandarin, and Spanish for international students
- **Expanded intent database** — Auto-scraped from UGM website changes on a weekly cadence
- **Better Study Mode retrieval** — Switch from TF-IDF to dense embeddings (sentence-transformers) for more accurate chunk retrieval
- **Persistent session history** — Reload previous conversations across browser sessions
- **Admin dashboard** — Analytics view showing most-asked questions, intent match rates, and LLM fallback rate
- **Timetable integration** — "What lectures do I have tomorrow?" powered by Moodle/SITS integration
- **Course recommendation** — "I'm interested in AI and data science, what should I study?"
- **Contextual document memory** — Remember uploaded documents across sessions so students don't have to re-upload
- **Handwritten notes OCR** — Upload a photo of lecture notes, get it OCR'd and indexed

---

## Author

**Collins Lemeke**

ChatUGM was built to solve two problems I kept seeing at University of Greater Manchester: students asking the same admin questions over and over with no fast way to get answers, and students struggling to get help with readings and assignments outside of office hours. One chatbot, two use cases, and a genuinely helpful tool that runs free on Hugging Face Spaces.

For questions, feedback, or feature requests, open a GitHub issue or reach out via Hugging Face.

---

## License

MIT License. Free to use, modify, and distribute. See [LICENSE](LICENSE) for full terms.

---

> *Built with Gradio, Hugging Face Transformers, Llama 3.2, and a lot of attention to how students actually ask questions. Meet Axiom. Chat about UGM. Upload your notes. Get help.*
