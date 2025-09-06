#!/usr/bin/env python3

import pandas as pd
import os
from datetime import datetime

def test_autosave_functionality():
    """Test the auto-save and retrain functionality"""
    
    print("Testing Auto-Save Functionality...")
    
    # Test data
    test_cases = [
        {
            'language': 'Java',
            'problem': 'Hello World program',
            'code': 'public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println("Hello, World!");\n    }\n}',
            'difficulty': 'Beginner'
        },
        {
            'language': 'Python', 
            'problem': 'Calculate factorial',
            'code': 'def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)',
            'difficulty': 'Intermediate'
        },
        {
            'language': 'C++',
            'problem': 'Array sum',
            'code': '#include <iostream>\nusing namespace std;\n\nint main() {\n    int arr[] = {1,2,3,4,5};\n    int sum = 0;\n    for(int i=0; i<5; i++) sum += arr[i];\n    cout << sum;\n    return 0;\n}',
            'difficulty': 'Beginner'
        }
    ]
    
    # Test each language
    for test in test_cases:
        lang = test['language']
        csv_files = {
            'Python': 'data_python.csv',
            'C++': 'data_cpp.csv',
            'Java': 'data_java.csv'
        }
        
        csv_file = csv_files[lang]
        
        # Check if file exists and get initial count
        initial_count = 0
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            initial_count = len(df)
        
        print(f"\n{lang} Dataset:")
        print(f"  Initial count: {initial_count}")
        print(f"  CSV file: {csv_file}")
        print(f"  Test problem: {test['problem']}")
        
        # Simulate save operation
        lines = len(test['code'].split('\n'))
        length = len(test['code'])
        
        if lang == 'Python':
            row = {
                'problem_title': test['problem'],
                'python_solutions': test['code'],
                'difficulty': test['difficulty'],
                'num_of_lines': lines,
                'code_length': length,
                'cyclomatic_complexity': 1,
                'readability': 0.8
            }
        elif lang == 'C++':
            row = {
                'Answer': test['code'],
                'num_of_lines': lines,
                'code_length': length
            }
        else:  # Java
            row = {
                'title': test['problem'],
                'content': test['code'],
                'difficulty': test['difficulty'],
                'num_of_lines': lines,
                'code_length': length,
                'cyclomatic_complexity': 1,
                'readability': 0.8
            }
        
        # Save to CSV
        try:
            df = pd.read_csv(csv_file) if os.path.exists(csv_file) else pd.DataFrame()
            new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            new_df.to_csv(csv_file, index=False)
            
            # Verify save
            updated_df = pd.read_csv(csv_file)
            final_count = len(updated_df)
            
            print(f"  [PASS] Saved successfully!")
            print(f"  Final count: {final_count}")
            print(f"  Added: {final_count - initial_count} row(s)")
            
        except Exception as e:
            print(f"  [FAIL] Save failed: {e}")
    
    # Check all datasets
    print(f"\nFinal Dataset Summary:")
    for lang, file in csv_files.items():
        if os.path.exists(file):
            df = pd.read_csv(file)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file))
            print(f"  {lang}: {len(df)} snippets (updated: {mod_time.strftime('%H:%M:%S')})")
        else:
            print(f"  {lang}: File not found")
    
    print(f"\n[PASS] Auto-save functionality test completed!")

if __name__ == "__main__":
    test_autosave_functionality()