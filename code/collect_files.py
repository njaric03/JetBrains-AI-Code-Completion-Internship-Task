import os

def collect_python_files(repo_path):
    python_files = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

repo_path = '../../crypto-lstm-predictor'
python_files = collect_python_files(repo_path)
print(f"Collected {len(python_files)} files.")