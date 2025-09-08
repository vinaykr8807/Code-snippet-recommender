import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def test_java_generation():
    """Test Java code generation with Ollama"""
    try:
        ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        ollama_model = os.getenv('OLLAMA_MODEL', 'gemma3:latest')
        
        # Java-specific prompt
        prompt = """You are an expert Java programmer. Write clean, efficient, and well-commented Java code.
Language: Java
Problem: Write a function to reverse a string

Requirements:
1. Write complete, runnable Java code
2. Use proper Java syntax and conventions
3. Include brief comments explaining key parts
4. Assess difficulty level (Beginner/Intermediate/Advanced)
5. Format: [DIFFICULTY: level] followed by code

Example format:
[DIFFICULTY: Intermediate]
public class Solution {
    public static void main(String[] args) {
    }
}

Generate Java solution:"""

        url = f"{ollama_host}/api/generate"
        headers = {'Content-Type': 'application/json'}
        payload = {
            'model': ollama_model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'num_predict': 300,
                'temperature': 0.2,
                'top_p': 0.9
            }
        }

        print("Testing Java code generation...")
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            generated_code = result.get('response', '')
            print("\n" + "="*50)
            print("GENERATED JAVA CODE:")
            print("="*50)
            print(generated_code)
            print("="*50)
            return True
        else:
            print(f"[ERROR] Generation failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Java generation test failed: {e}")
        return False

if __name__ == "__main__":
    test_java_generation()