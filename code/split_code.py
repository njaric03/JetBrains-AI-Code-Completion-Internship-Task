import json
import random

from collect_files import collect_python_files


def split_code(file_path, num_splits=5):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    num_lines = len(lines)
    splits = []

    for _ in range(num_splits):
        if num_lines < 3:
            continue
        split_line = random.randint(1, num_lines - 2)
        prefix = ''.join(lines[:split_line])
        middle = lines[split_line].strip()
        suffix = ''.join(lines[split_line + 1:])
        file_name = file_path.split('\\')[-1]
        if len(middle) > 0:
            splits.append({
                'file_name': file_name,
                'prefix': prefix,
                'middle': middle,
                'suffix': suffix
            })
    return splits

def generate_dataset(python_files, target_num):
    dataset = []
    for file in python_files:
        splits = split_code(file, num_splits=5)
        dataset.extend(splits)
        if len(dataset) >= target_num:
            break
    return dataset[:target_num]

def save_dataset(dataset, output_path):
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=4)


if __name__ == "__main__":

    repo_path = '../../crypto-lstm-predictor'
    python_files = collect_python_files(repo_path)
    dataset = generate_dataset(python_files, target_num=25)

    output_path = '../data/code_completion_dataset.json'
    save_dataset(dataset, output_path)
    print(f'Generated {len(dataset)} examples and saved to {output_path}')
