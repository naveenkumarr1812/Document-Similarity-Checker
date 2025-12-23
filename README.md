# Document Similarity Using Semantic Analysis

A Python application that leverages AI and machine learning to find similar documents based on semantic similarity using sentence embeddings.

## Overview

This project uses the Hugging Face Inference API with the `sentence-transformers/all-MiniLM-L6-v2` model to calculate semantic similarity between documents. It takes a query and compares it against a collection of documents to find the most relevant match using cosine similarity metrics.

## Features

- **Semantic Search**: Uses advanced sentence transformers for intelligent document matching
- **Multiple LLM Integrations**: Support for OpenAI, Anthropic, Google Gemini, and Hugging Face models
- **LangChain Integration**: Built with LangChain for flexible AI workflow management
- **Cosine Similarity**: Calculates similarity scores between documents and queries
- **Easy Configuration**: Environment variable-based configuration for secure API key management

## Prerequisites

- Python 3.8 or higher
- Hugging Face API token ([Get one here](https://huggingface.co/settings/tokens))

## Installation

1. Clone the repository:
```bash
git clone https://github.com/naveenkumarr1812/Document-Similarity-Checker.git
cd document-similarity-checker
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Create a `.env` file in the project root and add your Hugging Face API token:
```
HF_TOKEN=your_huggingface_api_token_here
```

## Usage

Run the script:
```bash
python document_similarity.py
```

### Example Output

```
Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.
```

## How It Works

1. **Document Collection**: Pre-defined documents are stored (e.g., cricket player descriptions)
2. **Query Input**: Provide a query string (e.g., "tell me about virat kohli")
3. **Embedding Generation**: Both documents and query are converted to embeddings using the sentence transformer model
4. **Similarity Calculation**: Cosine similarity is calculated between the query and all documents
5. **Result**: The document with the highest similarity score is returned

## Project Structure

```
document-similarity/
├── document_similarity.py    # Main application script
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not tracked by git)
├── .gitignore               # Git ignore rules
├── venv/                    # Virtual environment directory (not tracked by git)
└── README.md                # This file
```

## Dependencies

- **LangChain & LangChain Core**: AI framework for building applications
- **OpenAI Integration**: `langchain-openai`, `openai`
- **Anthropic Integration**: `langchain-anthropic`
- **Google Gemini Integration**: `langchain-google-genai`, `google-generativeai`
- **Hugging Face Integration**: `langchain-huggingface`, `transformers`, `huggingface-hub`
- **ML Utilities**: `numpy`, `scikit-learn`
- **Environment Management**: `python-dotenv`

For detailed versions, see [requirements.txt](requirements.txt)

## Configuration

### Environment Variables

- `HF_TOKEN`: Your Hugging Face API token (required)

Create a `.env` file:
```env
HF_TOKEN=hf_xxxxxxxxxxxxx
```



---

**Note**: Make sure to keep your `HF_TOKEN` secure and never commit the `.env` file to version control.
