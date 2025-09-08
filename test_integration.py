#!/usr/bin/env python3

import os
import sys
import requests
import json
from dotenv import load_dotenv

def test_java_integration():
    """Test Java code generation integration"""
    load_dotenv()
    
    # Test Ollama connection
    ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    ollama_model = os.getenv('OLLAMA_MODEL', 'gemma3:latest')
    
    print("Testing Java Integration...")
    print(f"Ollama Host: {ollama_host}")
    print(f"Ollama Model: {ollama_model}")
    
    # Test prompt
    query = "binary search algorithm"
    language = "Java"
    
    prompt = f"""You are an expert {language} programmer. Write clean, efficient, and well-commented {language} code.
Language: {language}
Problem: {query}

Requirements:
1. Write complete, runnable {language} code
2. Use proper {language} syntax and conventions
3. Include brief comments explaining key parts
4. Assess difficulty level (Beginner/Intermediate/Advanced)
5. Format: [DIFFICULTY: level] followed by code

Example format:
[DIFFICULTY: Intermediate]
public class Solution {{
    public static void main(String[] args) {{
    }}
}}

Generate {language} solution:"""

    try:
        url = f"{ollama_host}/api/generate"
        headers = {'Content-Type': 'application/json'}
        payload = {
            'model': ollama_model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'num_predict': 512,
                'temperature': 0.2,
                'top_p': 0.9,
                'repeat_penalty': 1.1
            }
        }

        print("\nSending request to Ollama...")
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        
        if r.status_code == 200:
            response = r.json()
            generated_code = response.get('response', '')
            
            print("[PASS] Java Integration Test PASSED")
            print(f"Generated {len(generated_code)} characters of Java code")
            print("\nGenerated Code Preview:")
            print("-" * 50)
            print(generated_code[:300] + "..." if len(generated_code) > 300 else generated_code)
            print("-" * 50)
            
            # Check for difficulty assessment
            if '[DIFFICULTY:' in generated_code:
                print("[PASS] Difficulty assessment included")
            else:
                print("[WARN] Difficulty assessment missing")
                
            return True
        else:
            print(f"[FAIL] Ollama request failed: {r.status_code}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Integration test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_java_integration()
    sys.exit(0 if success else 1)