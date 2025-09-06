# Code Snippet Recommendation System

A RAG (Retrieval-Augmented Generation) system that uses code snippets from CSV datasets to train and generate code recommendations using Ollama and FAISS.

## Features

- Loads code snippets from Python and C++ CSV datasets
- Creates embeddings using sentence-transformers
- Builds FAISS index for fast similarity search
- Generates code using Ollama (gemma3 model)
- Environment-based configuration

## Setup

### 1. Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Then edit `.env` with your actual values:

```
GITHUB_TOKEN=your_github_token_here
STACK_KEY=your_stack_overflow_key_here
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=gemma3:latest
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Test Setup

```bash
python test_setup.py
```

### 4. Run the Main Script

```bash
python jupyter_notebook_llm_code_similarity.py
```

## Dataset

- **Python CSV**: 1,681 code snippets with problem titles and solutions
- **C++ CSV**: 1,504 code snippets with answers

## Usage

### Option 1: Streamlit Web Application (Recommended)

Launch the interactive web interface:

```bash
python run_app.py
```

Or directly with Streamlit:

```bash
streamlit run streamlit_app.py
```

The web app provides:
- 🔍 Interactive code snippet search
- 📊 Model performance visualizations
- 🤖 AI code generation with Ollama
- 📈 Dataset analytics and metrics
- 🎨 Modern dark theme UI

### Option 2: Jupyter Notebook

Run the original notebook script:

```bash
python jupyter_notebook_llm_code_similarity.py
```

The main script will:

1. Load CSV datasets
2. Preprocess and normalize code snippets
3. Create embeddings using sentence-transformers
4. Build FAISS index for similarity search
5. Test Ollama connection and generation

## Files

- `streamlit_app.py` - **Main Streamlit web application**
- `run_app.py` - Application launcher script
- `jupyter_notebook_llm_code_similarity.py` - Original notebook script with RAG implementation
- `requirements.txt` - Python dependencies
- `.env` - Environment variables
- `data_python.csv` - Python code snippets dataset
- `data_cpp.csv` - C++ code snippets dataset

## Requirements

- Python 3.8+
- Ollama running locally with gemma3 model
- Internet connection for initial setup

## Notes

- The system uses RAG (Retrieval-Augmented Generation) rather than fine-tuning
- Embeddings are created using the `all-MiniLM-L6-v2` model
- FAISS is used for fast similarity search
- Ollama gemma3 model is used for code generation
