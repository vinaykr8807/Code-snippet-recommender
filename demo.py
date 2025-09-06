#!/usr/bin/env python3
"""
Demo script for the Code Snippet Recommendation System
This script demonstrates the core functionality without the web interface
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import requests
import json

def demo_search():
    """Demonstrate the search functionality"""
    print("🔍 CodeRec - Code Snippet Recommendation System Demo")
    print("=" * 60)
    
    # Load environment
    load_dotenv()
    
    # Load data
    print("📊 Loading dataset...")
    df_python = pd.read_csv('data_python.csv')
    
    # Prepare snippets
    python_snippets = []
    for idx, row in df_python.iterrows():
        if pd.notna(row['python_solutions']) and pd.notna(row['problem_title']):
            python_snippets.append({
                'title': row['problem_title'],
                'content': row['python_solutions'],
                'difficulty': row.get('difficulty', 'Unknown'),
                'num_of_lines': row.get('num_of_lines', 0)
            })
    
    df = pd.DataFrame(python_snippets)
    print(f"✅ Loaded {len(df)} Python code snippets")
    
    # Load model
    print("🤖 Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings
    print("🔄 Creating embeddings...")
    texts = (df['title'] + '\n' + df['content']).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Build FAISS index
    print("🔍 Building FAISS index...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Demo search
    print("\n" + "=" * 60)
    print("🎯 DEMO SEARCH")
    print("=" * 60)
    
    queries = [
        "sort array",
        "binary search",
        "merge sort",
        "two sum problem",
        "reverse linked list"
    ]
    
    for query in queries:
        print(f"\n🔍 Searching for: '{query}'")
        print("-" * 40)
        
        # Encode query
        q_emb = model.encode([query])
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        
        # Search
        D, I = index.search(q_emb.astype('float32'), 3)
        
        for i, (idx, score) in enumerate(zip(I[0], D[0])):
            if idx < len(df):
                result = df.iloc[idx]
                print(f"{i+1}. {result['title']} (Score: {score:.3f})")
                print(f"   Difficulty: {result['difficulty']}, Lines: {result['num_of_lines']}")
                print(f"   Code preview: {result['content'][:100]}...")
                print()
    
    # Test Ollama connection
    print("🤖 Testing Ollama connection...")
    ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    ollama_model = os.getenv('OLLAMA_MODEL', 'gemma3:latest')
    
    try:
        url = f"{ollama_host}/api/tags"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            models = r.json().get('models', [])
            model_names = [m['name'] for m in models]
            print(f"✅ Ollama connection successful")
            print(f"📋 Available models: {model_names}")
            if ollama_model in model_names:
                print(f"✅ Target model '{ollama_model}' is available")
            else:
                print(f"⚠️  Target model '{ollama_model}' not found")
        else:
            print(f"❌ Ollama connection failed: {r.status_code}")
    except Exception as e:
        print(f"❌ Ollama connection error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed!")
    print("💡 To use the full web interface, run: python run_app.py")
    print("=" * 60)

if __name__ == "__main__":
    demo_search()
