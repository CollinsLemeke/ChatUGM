---
title: UGM Chatbot
emoji: 🎓
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
license: mit
---

# UGM Chatbot - University of Greater Manchester

An intelligent chatbot assistant for the University of Greater Manchester with Study Mode capabilities.

## Features

### 💬 Normal Chat Mode
- Answer questions about the university (courses, fees, admissions, etc.)
- Intent-based matching with TF-IDF similarity
- LLM fallback using Llama-3.2-1B-Instruct

### 📚 Study Mode
Upload your study materials and get help:
- **Interact Mode**: Ask questions about your uploaded documents
- **Assignment Mode**: Get help understanding and completing assignments
- **Summary Mode**: Get comprehensive summaries with key points

### Supported File Types
- PDF (.pdf)
- Word Documents (.docx, .doc)
- PowerPoint (.pptx, .ppt)
- Excel (.xlsx, .xls)
- CSV (.csv)
- JSON (.json)
- Text (.txt)

## Setup

1. Clone this repository
2. Add your `HF_TOKEN` as a secret in your Hugging Face Space settings
3. Replace `intents.json` with your actual dataset
4. Add your `logo.png` file to the repository
5. Deploy!

## Environment Variables

- `HF_TOKEN`: Your Hugging Face token (required for accessing Llama model)

## Files Structure

```
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── intents.json        # Intent dataset (replace with your own)
├── logo.png            # University logo
└── README.md           # This file
```

## Usage

1. **Normal Chat**: Just type your question about the university
2. **Study Mode**: 
   - Click the 📚 button to enable Study Mode
   - Upload a document using the 📎 button
   - Type `interact`, `assignment`, or `summary` to choose your mode
   - Ask questions about your document
   - Type `exit` to leave the current mode

## License

MIT License