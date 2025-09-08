import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
from dotenv import load_dotenv
import requests
from sentence_transformers import SentenceTransformer
import faiss
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import csv
from datetime import datetime

# Initialize session state for auto-save
if 'auto_save_enabled' not in st.session_state:
    st.session_state.auto_save_enabled = True

# GitHub and StackOverflow API helpers
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
STACK_KEY = os.getenv('STACK_KEY')
HEADERS = {'Accept': 'application/vnd.github.v3+json'}
if GITHUB_TOKEN:
    HEADERS['Authorization'] = f'token {GITHUB_TOKEN}'

def fetch_github_code_snippets(query, language=None, max_files=200):
    """Fetch code snippets from GitHub"""
    items = []
    per_page = 30
    page = 1
    fetched = 0
    q = query
    if language:
        q += f' language:{language}'

    while fetched < max_files:
        url = f'https://api.github.com/search/code?q={requests.utils.quote(q)}&per_page={per_page}&page={page}'
        r = requests.get(url, headers=HEADERS)
        if r.status_code != 200:
            break
        resp = r.json()
        results = resp.get('items', [])
        if not results:
            break
        for item in results:
            if fetched >= max_files:
                break
            try:
                download_url = item.get('html_url').replace('/blob/', '/')
                cr = requests.get(download_url, headers=HEADERS, timeout=10)
                if cr.status_code == 200:
                    content = cr.text
                    items.append({
                        'source': 'github',
                        'repo': item.get('repository', {}).get('full_name'),
                        'path': item.get('path'),
                        'url': item.get('html_url'),
                        'content': content,
                        'language': item.get('language') or language
                    })
                    fetched += 1
            except:
                continue
        page += 1
        time.sleep(1)
    return items

def fetch_stackoverflow_code_snippets(tag=None, pagesize=100, max_pages=5):
    """Fetch code snippets from StackOverflow"""
    base = 'https://api.stackexchange.com/2.3'
    page = 1
    snippets = []
    while page <= max_pages:
        params = {
            'order': 'desc',
            'sort': 'activity',
            'site': 'stackoverflow',
            'pagesize': pagesize,
            'page': page,
            'filter': 'withbody'
        }
        if STACK_KEY:
            params['key'] = STACK_KEY
        if tag:
            params['tagged'] = tag
        url = f"{base}/questions"
        r = requests.get(url, params=params)
        if r.status_code != 200:
            break
        data = r.json()
        for q in data.get('items', []):
            question_id = q['question_id']
            title = q.get('title')
            link = q.get('link')
            ar = requests.get(f"{base}/questions/{question_id}/answers", params={'order':'desc','sort':'votes','site':'stackoverflow','filter':'withbody', **({'key': STACK_KEY} if STACK_KEY else {})})
            if ar.status_code != 200:
                continue
            answers = ar.json().get('items', [])
            for a in answers:
                body = a.get('body', '')
                soup = BeautifulSoup(body, 'html.parser')
                code_blocks = [c.get_text() for c in soup.find_all('code')]
                for cb in code_blocks:
                    snippets.append({
                        'source':'stackoverflow',
                        'question_id': question_id,
                        'answer_id': a.get('answer_id'),
                        'title': title,
                        'link': link,
                        'content': cb
                    })
        if not data.get('has_more'):
            break
        page += 1
        time.sleep(1)
    return snippets

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="CodeRec - Code Snippet Recommendation System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
dark_css = """
<style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stApp {
        background-color: #0e1117;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: white;
        border: 1px solid #4a4a4a;
    }
    .stSelectbox > div > div > select {
        background-color: #262730;
        color: white;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #4a4a4a;
        margin: 0.5rem 0;
    }
    .code-snippet {
        background-color: #1e1e1e;
        border: 1px solid #4a4a4a;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
    .similarity-score {
        color: #00d4aa;
        font-weight: bold;
    }
    .header-section {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        text-align: center;
    }
</style>
"""

# Custom CSS for light theme
light_css = """
<style>
    .main {
        background-color: #ffffff;
        color: #000000 !important;
    }
    .stApp {
        background-color: #ffffff;
        color: #000000 !important;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: #000000 !important;
        border: 1px solid #cccccc;
    }
    .stSelectbox > div > div > select {
        background-color: #ffffff;
        color: #000000 !important;
        border: 1px solid #cccccc;
    }
    .stMarkdown, .stText, p, div, span {
        color: #000000 !important;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        color: #000000 !important;
    }
    .code-snippet {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        color: #000000 !important;
    }
    .similarity-score {
        color: #007bff !important;
        font-weight: bold;
    }
    .header-section {
        background: linear-gradient(90deg, #007bff 0%, #28a745 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        text-align: center;
    }
</style>
"""

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        all_snippets = []
        
        # Load Python dataset
        if os.path.exists('data_python.csv'):
            df_python = pd.read_csv('data_python.csv')
            for idx, row in df_python.iterrows():
                if pd.notna(row['python_solutions']) and pd.notna(row['problem_title']):
                    all_snippets.append({
                        'source': 'python_csv',
                        'title': row['problem_title'],
                        'content': row['python_solutions'],
                        'language': 'python',
                        'difficulty': row.get('difficulty', 'Unknown'),
                        'num_of_lines': row.get('num_of_lines', 0),
                        'code_length': row.get('code_length', 0),
                        'cyclomatic_complexity': row.get('cyclomatic_complexity', 0),
                        'readability': row.get('readability', 0)
                    })
        
        # Load C++ dataset
        if os.path.exists('data_cpp.csv'):
            df_cpp = pd.read_csv('data_cpp.csv')
            for idx, row in df_cpp.iterrows():
                if pd.notna(row['Answer']):
                    all_snippets.append({
                        'source': 'cpp_csv',
                        'title': f'C++ Solution {idx}',
                        'content': row['Answer'],
                        'language': 'cpp',
                        'difficulty': 'Unknown',
                        'num_of_lines': row.get('num_of_lines', 0),
                        'code_length': row.get('code_length', 0),
                        'cyclomatic_complexity': 0,
                        'readability': 0
                    })
        
        # Load Java dataset if exists
        if os.path.exists('data_java.csv'):
            df_java = pd.read_csv('data_java.csv')
            for idx, row in df_java.iterrows():
                if pd.notna(row.get('content')):
                    all_snippets.append({
                        'source': 'java_csv',
                        'title': row.get('title', f'Java Solution {idx}'),
                        'content': row['content'],
                        'language': 'java',
                        'difficulty': 'Unknown',
                        'num_of_lines': 0,
                        'code_length': 0,
                        'cyclomatic_complexity': 0,
                        'readability': 0
                    })
        
        df = pd.DataFrame(all_snippets)
        
        # Normalize code
        def normalize_code(text):
            if not isinstance(text, str):
                return ''
            text = text.strip('\n')
            text = re.sub(r"\n\s*\n+", '\n\n', text)
            return text
        
        df['code'] = df['content'].astype(str).apply(normalize_code)
        df['length_chars'] = df['code'].str.len()
        
        # Filter reasonable sized snippets
        df = df[(df['length_chars'] > 20) & (df['length_chars'] < 20000)].reset_index(drop=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_model():
    """Load the sentence transformer model"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_embeddings_and_index():
    """Load precomputed embeddings and FAISS index"""
    try:
        if os.path.exists('embeddings.npy') and os.path.exists('code_faiss.index'):
            embeddings = np.load('embeddings.npy')
            index = faiss.read_index('code_faiss.index')
            return embeddings, index
        else:
            return None, None
    except Exception as e:
        st.error(f"Error loading embeddings/index: {e}")
        return None, None

def create_embeddings_and_index(df, model):
    """Create embeddings and FAISS index"""
    try:
        # Create text for embedding
        texts = (df.get('title', '').fillna('') + '\n' + df['code']).tolist()
        
        # Compute embeddings in batches
        batch_size = 32
        embeddings = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            emb = model.encode(batch, show_progress_bar=False)
            embeddings.append(emb)
            progress_value = min(1.0, (i + batch_size) / len(texts))
            progress_bar.progress(progress_value)
            status_text.text(f'Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}')
        
        embeddings = np.vstack(embeddings)
        
        # Build FAISS index
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        # Save embeddings and index
        np.save('embeddings.npy', embeddings)
        faiss.write_index(index, 'code_faiss.index')
        
        progress_bar.empty()
        status_text.empty()
        
        return embeddings, index
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return None, None

def retrieve_similar_snippets(query_text, top_k=5, language_filter=None):
    """Retrieve similar code snippets"""
    if st.session_state.model is None or st.session_state.index is None:
        return []

    try:
        q_emb = st.session_state.model.encode([query_text])
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        D, I = st.session_state.index.search(q_emb.astype('float32'), top_k * 3)  # Get more to filter

        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < len(st.session_state.df):
                row = st.session_state.df.loc[idx]
                if language_filter and row['language'] != language_filter:
                    continue
                results.append({
                    'idx': int(idx),
                    'score': float(score),
                    'code': row['code'],
                    'title': row['title'],
                    'difficulty': row['difficulty'],
                    'num_of_lines': row['num_of_lines'],
                    'language': row['language'],
                    'meta': row.to_dict()
                })
                if len(results) >= top_k:
                    break
        return results
    except Exception as e:
        st.error(f"Error retrieving snippets: {e}")
        return []

def save_generated_code_to_csv(problem_statement, generated_code, language, difficulty="Unknown"):
    """Save generated code to appropriate CSV dataset"""
    try:
        # Calculate metrics
        lines = len(generated_code.split('\n'))
        length = len(generated_code)
        
        # Determine CSV file
        csv_files = {
            'Python': 'data_python.csv',
            'C++': 'data_cpp.csv', 
            'Java': 'data_java.csv'
        }
        
        csv_file = csv_files.get(language, 'data_python.csv')
        
        # Prepare row data
        if language == 'Python':
            row = {
                'problem_title': problem_statement,
                'python_solutions': generated_code,
                'difficulty': difficulty,
                'num_of_lines': lines,
                'code_length': length,
                'cyclomatic_complexity': 1,
                'readability': 0.8
            }
        elif language == 'C++':
            row = {
                'Answer': generated_code,
                'num_of_lines': lines,
                'code_length': length
            }
        else:  # Java
            row = {
                'title': problem_statement,
                'content': generated_code,
                'difficulty': difficulty,
                'num_of_lines': lines,
                'code_length': length,
                'cyclomatic_complexity': 1,
                'readability': 0.8
            }
        
        # Read existing data
        df = pd.read_csv(csv_file) if os.path.exists(csv_file) else pd.DataFrame()
        
        # Add new row
        new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        new_df.to_csv(csv_file, index=False)
        
        return True
    except Exception as e:
        st.error(f"Failed to save code: {e}")
        return False

def calculate_code_metrics(code):
    """Calculate code complexity metrics"""
    lines = len([line for line in code.split('\n') if line.strip()])
    chars = len(code)
    
    # Simple complexity indicators
    loops = code.count('for') + code.count('while') + code.count('do')
    conditions = code.count('if') + code.count('else') + code.count('switch')
    functions = code.count('def ') + code.count('function') + code.count('public ') + code.count('private ')
    
    # Estimate complexity score
    complexity = loops * 2 + conditions * 1.5 + functions * 1
    
    return {
        'lines': lines,
        'characters': chars,
        'loops': loops,
        'conditions': conditions,
        'functions': functions,
        'complexity_score': complexity
    }

def show_complexity_graphs(code, difficulty, language):
    """Show code complexity analysis graphs"""
    metrics = calculate_code_metrics(code)
    
    # Expected ranges for each difficulty
    difficulty_ranges = {
        'Easy': {'lines': (5, 20), 'complexity': (0, 5), 'functions': (1, 2)},
        'Intermediate': {'lines': (15, 50), 'complexity': (3, 15), 'functions': (2, 5)},
        'Advanced': {'lines': (30, 100), 'complexity': (10, 30), 'functions': (3, 10)}
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Lines of code comparison
        expected_range = difficulty_ranges[difficulty]['lines']
        fig_lines = go.Figure()
        fig_lines.add_trace(go.Bar(
            x=['Expected Min', 'Actual', 'Expected Max'],
            y=[expected_range[0], metrics['lines'], expected_range[1]],
            marker_color=['lightblue', 'darkblue', 'lightblue']
        ))
        fig_lines.update_layout(
            title=f"Lines of Code ({difficulty})",
            yaxis_title="Lines",
            height=300
        )
        st.plotly_chart(fig_lines, use_container_width=True)
    
    with col2:
        # Complexity score
        expected_range = difficulty_ranges[difficulty]['complexity']
        fig_complexity = go.Figure()
        fig_complexity.add_trace(go.Bar(
            x=['Expected Min', 'Actual', 'Expected Max'],
            y=[expected_range[0], metrics['complexity_score'], expected_range[1]],
            marker_color=['lightgreen', 'darkgreen', 'lightgreen']
        ))
        fig_complexity.update_layout(
            title=f"Complexity Score ({difficulty})",
            yaxis_title="Score",
            height=300
        )
        st.plotly_chart(fig_complexity, use_container_width=True)
    
    with col3:
        # Code structure breakdown
        fig_structure = go.Figure(data=[
            go.Bar(name='Loops', x=['Count'], y=[metrics['loops']]),
            go.Bar(name='Conditions', x=['Count'], y=[metrics['conditions']]),
            go.Bar(name='Functions', x=['Count'], y=[metrics['functions']])
        ])
        fig_structure.update_layout(
            title="Code Structure",
            yaxis_title="Count",
            height=300,
            barmode='group'
        )
        st.plotly_chart(fig_structure, use_container_width=True)
    
    # Metrics summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Lines of Code", metrics['lines'])
    with col2:
        st.metric("Complexity Score", f"{metrics['complexity_score']:.1f}")
    with col3:
        st.metric("Control Structures", metrics['loops'] + metrics['conditions'])
    with col4:
        # Difficulty match indicator
        expected_lines = difficulty_ranges[difficulty]['lines']
        expected_complexity = difficulty_ranges[difficulty]['complexity']
        
        lines_match = expected_lines[0] <= metrics['lines'] <= expected_lines[1]
        complexity_match = expected_complexity[0] <= metrics['complexity_score'] <= expected_complexity[1]
        
        if lines_match and complexity_match:
            st.metric("Difficulty Match", "Perfect")
        elif lines_match or complexity_match:
            st.metric("Difficulty Match", "Partial")
        else:
            st.metric("Difficulty Match", "Mismatch")

def retrain_model():
    """Retrain embeddings and FAISS index with updated data"""
    try:
        # Clear cached data
        if 'df' in st.session_state:
            st.session_state.df = None
        if 'embeddings' in st.session_state:
            st.session_state.embeddings = None
        if 'index' in st.session_state:
            st.session_state.index = None
            
        # Remove cached files
        for file in ['embeddings.npy', 'code_faiss.index']:
            if os.path.exists(file):
                os.remove(file)
                
        return True
    except Exception as e:
        st.error(f"Retrain failed: {e}")
        return False

def generate_code_with_ollama(problem_statement, retrieved_snippets, language, difficulty="Intermediate", max_tokens=512, temperature=0.2):
    """Generate code using Ollama with language-specific prompts and difficulty assessment"""
    try:
        # Language-specific syntax examples
        lang_examples = {
            'Python': 'def function_name():\n    pass',
            'C++': 'int main() {\n    return 0;\n}',
            'Java': 'public class Solution {\n    public static void main(String[] args) {\n    }\n}'
        }
        
        # Difficulty-specific instructions
        difficulty_instructions = {
            'Easy': 'Write simple, beginner-friendly code with extensive comments and basic algorithms.',
            'Intermediate': 'Write moderately complex code with good practices and efficient algorithms.',
            'Advanced': 'Write sophisticated code with advanced algorithms, design patterns, and optimizations.'
        }
        
        # Build enhanced prompt
        prompt_parts = [
            f"You are an expert {language} programmer. Write clean, efficient, and well-commented {language} code.",
            f"Language: {language}",
            f"Problem: {problem_statement}",
            f"Difficulty Level: {difficulty}",
            f"Instructions: {difficulty_instructions.get(difficulty, difficulty_instructions['Intermediate'])}",
            "",
            "Reference snippets for context:"
        ]
        
        # Add retrieved snippets with better formatting
        for i, s in enumerate(retrieved_snippets[:3]):  # Limit to top 3
            difficulty = s.get('difficulty', 'Unknown')
            prompt_parts.append(f"Reference {i+1} (similarity: {s['score']:.3f}, difficulty: {difficulty}):")
            prompt_parts.append(s['code'][:500] + ('...' if len(s['code']) > 500 else ''))
            prompt_parts.append("")
        
        prompt_parts.extend([
            f"Requirements:",
            f"1. Write complete, runnable {language} code at {difficulty} level",
            f"2. Use proper {language} syntax and conventions",
            f"3. Include appropriate comments for {difficulty} level",
            f"4. Must match requested difficulty: {difficulty}",
            f"5. Format: [DIFFICULTY: {difficulty}] followed by code",
            "",
            f"Example format:",
            f"[DIFFICULTY: Intermediate]",
            lang_examples.get(language, 'code here'),
            "",
            f"Generate {language} solution:"
        ])

        prompt = '\n'.join(prompt_parts)

        # Ollama API call
        ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        ollama_model = os.getenv('OLLAMA_MODEL', 'gemma3:latest')

        url = f"{ollama_host}/api/generate"
        headers = {'Content-Type': 'application/json'}
        payload = {
            'model': ollama_model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'num_predict': max_tokens,
                'temperature': temperature,
                'top_p': 0.9,
                'repeat_penalty': 1.1
            }
        }

        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f'Ollama generation failed: {r.status_code} {r.text[:200]}')

        response = r.json()
        if 'response' in response:
            return response['response']
        else:
            return str(response)
    except Exception as e:
        return f'Generation failed: {e}'

def create_performance_visualizations(df, theme="Dark", search_results=None, language="Python"):
    """Create model performance visualizations"""
    st.subheader("📊 Model Performance & Dataset Analytics")
    
    # Set theme colors
    font_color = 'white' if theme == 'Dark' else 'black'
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Difficulty distribution for current language
        difficulty_counts = df['difficulty'].value_counts()
        fig_difficulty = px.pie(
            values=difficulty_counts.values,
            names=difficulty_counts.index,
            title=f"{language} Problem Difficulty Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_difficulty.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=font_color
        )
        st.plotly_chart(fig_difficulty, use_container_width=True)
    
    with col2:
        # Code length distribution for current language
        fig_length = px.histogram(
            df, x='length_chars',
            title=f"{language} Code Length Distribution",
            nbins=30,
            color_discrete_sequence=['#00d4aa']
        )
        fig_length.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=font_color,
            xaxis_title="Code Length (characters)",
            yaxis_title="Count"
        )
        st.plotly_chart(fig_length, use_container_width=True)
    
    # Show search results visualization if available
    if search_results:
        col3, col4 = st.columns(2)
        
        with col3:
            # Similarity scores of search results
            scores = [r['score'] for r in search_results]
            titles = [r['title'][:30] + '...' if len(r['title']) > 30 else r['title'] for r in search_results]
            fig_similarity = px.bar(
                x=titles, y=scores,
                title="Search Results Similarity Scores",
                color=scores,
                color_continuous_scale='viridis'
            )
            fig_similarity.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=font_color,
                xaxis_title="Code Snippets",
                yaxis_title="Similarity Score",
                xaxis={'tickangle': 45}
            )
            st.plotly_chart(fig_similarity, use_container_width=True)
        
        with col4:
            # Lines of code in search results
            lines = [r['num_of_lines'] for r in search_results]
            fig_lines = px.scatter(
                x=range(len(search_results)), y=lines,
                title="Search Results - Lines of Code",
                color=scores,
                color_continuous_scale='plasma'
            )
            fig_lines.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=font_color,
                xaxis_title="Result Index",
                yaxis_title="Lines of Code"
            )
            st.plotly_chart(fig_lines, use_container_width=True)
    else:
        col3, col4 = st.columns(2)
        
        with col3:
            # Lines of code vs complexity for current language
            fig_complexity = px.scatter(
                df, x='num_of_lines', y='cyclomatic_complexity',
                title=f"{language} Lines vs Complexity",
                color='difficulty',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_complexity.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=font_color,
                xaxis_title="Number of Lines",
                yaxis_title="Cyclomatic Complexity"
            )
            st.plotly_chart(fig_complexity, use_container_width=True)
        
        with col4:
            # Readability distribution for current language
            fig_readability = px.box(
                df, y='readability',
                title=f"{language} Code Readability",
                color_discrete_sequence=['#3b82f6']
            )
            fig_readability.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=font_color,
                yaxis_title="Readability Score"
            )
            st.plotly_chart(fig_readability, use_container_width=True)

def main():
    # Theme selector
    theme = st.sidebar.selectbox("Theme", ["Dark", "Light"], index=0)
    if theme == "Dark":
        st.markdown(dark_css, unsafe_allow_html=True)
    else:
        st.markdown(light_css, unsafe_allow_html=True)

    # Header section
    header_color = "white" if theme == "Dark" else "black"
    header_bg = "#e0e0e0" if theme == "Dark" else "#f0f0f0"
    st.markdown(f"""
    <div class="header-section">
        <h1 style="color: {header_color}; margin: 0; font-size: 3rem;">🔍 CodeRec</h1>
        <h2 style="color: {header_color}; margin: 0.5rem 0; font-weight: normal;">Find the right code snippet fast</h2>
        <p style="color: {header_bg}; margin: 0;">Describe your problem and get tailored snippets from our datasets</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")

        # Language selection - always show all options
        available_languages = ["Python", "C++", "Java"]
        
        # Add snippet counts if data is available
        if st.session_state.df is not None:
            lang_counts = st.session_state.df['language'].value_counts()
            lang_map = {'python': 'Python', 'cpp': 'C++', 'java': 'Java'}
            display_languages = []
            for lang_code, display_name in lang_map.items():
                count = lang_counts.get(lang_code, 0)
                if count > 0:
                    display_languages.append(f"{display_name} ({count} snippets)")
                else:
                    display_languages.append(f"{display_name} (Generate with AI)")
            available_languages = display_languages
        
        language_selection = st.selectbox(
            "Programming Language",
            available_languages,
            index=0
        )
        
        # Extract just the language name
        language = language_selection.split(' (')[0]

        # Number of results
        num_results = st.slider("Number of Results", 1, 10, 5)

        # Similarity threshold
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.1, 0.05)

        # Model settings
        st.subheader("🤖 Model Settings")
        use_ollama = st.checkbox("Use Ollama for Code Generation", value=True)
        auto_save = st.checkbox("Auto-save Generated Code", value=True, help="Automatically save generated code to dataset and retrain")
        
        # Difficulty selection
        difficulty_level = st.selectbox(
            "Code Difficulty Level",
            ["Easy", "Intermediate", "Advanced"],
            index=1,
            help="Select complexity level for generated code"
        )
        
        if use_ollama:
            max_tokens = st.slider("Max Tokens", 100, 1000, 512)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
        
        # Manual retrain button
        if st.button("🔄 Retrain Model", help="Manually retrain with current datasets"):
            with st.spinner("Retraining model..."):
                if retrain_model():
                    st.success("Model retrained successfully!")
                    st.rerun()
                else:
                    st.error("Retrain failed")
    
    # Load data and model
    if st.session_state.df is None:
        with st.spinner("Loading dataset..."):
            st.session_state.df = load_data()

    if st.session_state.model is None:
        with st.spinner("Loading model..."):
            st.session_state.model = load_model()

    if st.session_state.embeddings is None or st.session_state.index is None:
        with st.spinner("Loading embeddings and index..."):
            st.session_state.embeddings, st.session_state.index = load_embeddings_and_index()

            # If no precomputed embeddings, create them
            if st.session_state.embeddings is None or st.session_state.index is None:
                st.info("Creating embeddings and FAISS index... This may take a few minutes.")
                st.session_state.embeddings, st.session_state.index = create_embeddings_and_index(
                    st.session_state.df, st.session_state.model
                )

    # Filter dataset by selected language
    filtered_df = pd.DataFrame()  # Default empty
    if st.session_state.df is not None:
        lang_map = {"Python": "python", "C++": "cpp", "Java": "java"}
        selected_lang = lang_map.get(language, "python")
        filtered_df = st.session_state.df[st.session_state.df['language'] == selected_lang]

        # Dataset metrics with last updated info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Snippets", len(filtered_df))
        with col2:
            if len(filtered_df) > 0:
                st.metric("Avg Code Length", f"{filtered_df['length_chars'].mean():.0f} chars")
            else:
                st.metric("Avg Code Length", "N/A")
        with col3:
            if len(filtered_df) > 0:
                st.metric("Avg Lines", f"{filtered_df['num_of_lines'].mean():.1f}")
            else:
                st.metric("Avg Lines", "N/A")
        with col4:
            # Show last modified time of dataset
            lang_files = {"Python": "data_python.csv", "C++": "data_cpp.csv", "Java": "data_java.csv"}
            csv_file = lang_files.get(language, "data_python.csv")
            if os.path.exists(csv_file):
                mod_time = datetime.fromtimestamp(os.path.getmtime(csv_file))
                st.metric("Last Updated", mod_time.strftime("%H:%M"))
            else:
                st.metric("Last Updated", "N/A")
    
    # Main search interface
    st.subheader("🔍 Search Code Snippets")
    
    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Describe your problem",
            placeholder="e.g., sort array, binary search, merge sort",
            key="search_input"
        )
    with col2:
        search_button = st.button("Get Recommendations", type="primary", use_container_width=True)
    
    # Initialize variables
    results = []
    filtered_results = []
    
    # Search results
    if search_button and query:
        st.markdown(f"**Showing results for \"{query}\" in {language}**")
        
        # Retrieve similar snippets
        with st.spinner("Searching for similar code snippets..."):
            lang_map = {"Python": "python", "C++": "cpp", "Java": "java"}
            selected_lang = lang_map.get(language, "python")
            results = retrieve_similar_snippets(query, top_k=num_results, language_filter=selected_lang)
        
        if results:
            # Filter by similarity threshold
            filtered_results = [r for r in results if r['score'] >= similarity_threshold]
            
            if filtered_results:
                # Display results
                for i, result in enumerate(filtered_results):
                    with st.expander(f"📝 {result['title']} (Similarity: {result['score']:.3f})", expanded=True):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            # Use actual language from result data
                            result_lang = result.get('language', 'python')
                            code_lang = {'python': 'python', 'cpp': 'cpp', 'java': 'java'}.get(result_lang, 'python')
                            st.code(result['code'], language=code_lang)
                        
                        with col2:
                            st.metric("Similarity", f"{result['score']:.3f}")
                            st.metric("Lines", result['num_of_lines'])
                            st.metric("Difficulty", result['difficulty'])
                            
                            # Copy button
                            if st.button(f"📋 Copy Code", key=f"copy_{i}"):
                                result_lang = result.get('language', 'python')
                                code_lang = {'python': 'python', 'cpp': 'cpp', 'java': 'java'}.get(result_lang, 'python')
                                st.code(result['code'], language=code_lang)
                                st.success("Code copied to clipboard!")
        
    # Always show AI generation option (even if no results found)
    if use_ollama and query:
        st.subheader("🤖 AI-Generated Solution")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            generate_btn = st.button("Generate Code with AI", type="secondary")
        with col2:
            st.info(f"Target: {language}")
        
        if generate_btn:
            with st.spinner(f"Generating {difficulty_level} {language} code with Ollama..."):
                # Use empty results if no snippets found
                reference_snippets = filtered_results[:3] if filtered_results else []
                generated_code = generate_code_with_ollama(query, reference_snippets, language, difficulty_level, max_tokens, temperature)
                
            # Extract difficulty if present
            difficulty_match = re.search(r'\[DIFFICULTY:\s*(\w+)\]', generated_code)
            if difficulty_match:
                difficulty = difficulty_match.group(1)
                generated_code = re.sub(r'\[DIFFICULTY:\s*\w+\]\s*', '', generated_code)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Generated {language} Solution:**")
                with col2:
                    # Color code difficulty
                    diff_colors = {'Beginner': 'green', 'Intermediate': 'orange', 'Advanced': 'red'}
                    color = diff_colors.get(difficulty, 'blue')
                    st.markdown(f"**Difficulty:** <span style='color:{color}'>{difficulty}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"**Generated {language} Solution:**")
            
            code_lang = {'Python': 'python', 'C++': 'cpp', 'Java': 'java'}.get(language, 'python')
            st.code(generated_code.strip(), language=code_lang)
            
            # Show complexity analysis
            st.subheader("📊 Code Complexity Analysis")
            show_complexity_graphs(generated_code.strip(), difficulty_level, language)
            
            # Auto-save if enabled
            if auto_save:
                difficulty = difficulty_match.group(1) if difficulty_match else difficulty_level
                if save_generated_code_to_csv(query, generated_code.strip(), language, difficulty):
                    st.success(f"✅ Auto-saved to {language} dataset & retrained!")
                    # Auto-retrain
                    retrain_model()
            
            # Manual save and copy buttons
            col1, col2 = st.columns(2)
            with col1:
                if not auto_save and st.button("💾 Save to Dataset", key="save_generated"):
                    difficulty = difficulty_match.group(1) if difficulty_match else difficulty_level
                    if save_generated_code_to_csv(query, generated_code.strip(), language, difficulty):
                        st.success(f"Code saved to {language} dataset!")
                        with st.spinner("Retraining model..."):
                            if retrain_model():
                                st.success("Model retrained!")
                                st.rerun()
            with col2:
                if st.button("📋 Copy Generated Code", key="copy_generated"):
                    st.success("Generated code ready to copy!")
        
        # Show messages about search results
        if results:
            if not filtered_results:
                st.warning(f"No results found above similarity threshold of {similarity_threshold}")
        else:
            if len(filtered_df) == 0:
                st.info(f"No {language} snippets in dataset. Use AI generation above to create code.")
            else:
                st.error("No similar code snippets found. Try a different query or use AI generation.")
    
    # Show info for languages without datasets
    elif not search_button and len(filtered_df) == 0:
        st.info(f"No {language} snippets in dataset. Use AI generation above to create code.")
    
    # Performance visualizations
    if st.session_state.df is not None:
        # Get current search results if available
        current_results = None
        if 'search_button' in locals() and search_button and query and 'filtered_results' in locals():
            current_results = filtered_results
        
        create_performance_visualizations(filtered_df if 'filtered_df' in locals() else st.session_state.df, theme, current_results, language)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p>CodeRec - Code Snippet Recommendation System</p>
            <p>Powered by Sentence Transformers, FAISS, and Ollama</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
