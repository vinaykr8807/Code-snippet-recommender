import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def test_ollama():
    """Test Ollama connection"""
    try:
        host = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
        model = os.getenv('OLLAMA_MODEL', 'gemma3:latest')
        
        # Test connection
        response = requests.get(f"{host}/api/tags", timeout=5)
        if response.status_code == 200:
            print("[OK] Ollama server is running")
            
            # Test generation
            payload = {
                'model': model,
                'prompt': 'Write a simple Python hello world function',
                'stream': False,
                'options': {'num_predict': 50}
            }
            
            gen_response = requests.post(f"{host}/api/generate", 
                                       json=payload, timeout=30)
            if gen_response.status_code == 200:
                print("[OK] Ollama generation working")
                return True
            else:
                print(f"[ERROR] Ollama generation failed: {gen_response.status_code}")
        else:
            print(f"[ERROR] Ollama server not responding: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Ollama error: {e}")
    return False

def test_github_api():
    """Test GitHub API"""
    try:
        token = os.getenv('GITHUB_TOKEN')
        headers = {'Accept': 'application/vnd.github.v3+json'}
        if token:
            headers['Authorization'] = f'token {token}'
        
        response = requests.get('https://api.github.com/rate_limit', 
                              headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            remaining = data['rate']['remaining']
            print(f"[OK] GitHub API working - {remaining} requests remaining")
            return True
        else:
            print(f"[ERROR] GitHub API failed: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] GitHub API error: {e}")
    return False

def test_stackoverflow_api():
    """Test StackOverflow API"""
    try:
        key = os.getenv('STACK_KEY')
        params = {'site': 'stackoverflow', 'pagesize': 1}
        if key:
            params['key'] = key
        
        response = requests.get('https://api.stackexchange.com/2.3/questions',
                              params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            quota = data.get('quota_remaining', 'Unknown')
            print(f"[OK] StackOverflow API working - {quota} quota remaining")
            return True
        else:
            print(f"[ERROR] StackOverflow API failed: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] StackOverflow API error: {e}")
    return False

if __name__ == "__main__":
    print("Testing APIs and Ollama...")
    print("-" * 40)
    
    ollama_ok = test_ollama()
    github_ok = test_github_api()
    stack_ok = test_stackoverflow_api()
    
    print("-" * 40)
    if ollama_ok and github_ok and stack_ok:
        print("[OK] All services working!")
    else:
        print("[ERROR] Some services have issues")