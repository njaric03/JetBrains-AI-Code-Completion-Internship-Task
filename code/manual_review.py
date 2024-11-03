import json


def manual_review(dataset):
    for i, example in enumerate(dataset):
        print(f"Example {i + 1}/{len(dataset)}")
        print("File name:", example['file_name'])
        print("Model completion:", example['model_completion'])
        print("Actual middle:", example['middle'])

        label = input("Is the model completion incorrect, partially correct, or correct? (0/0.5/1): ")

        if label == '0':
            example['Label'] = 'incorrect'
        elif label == '0.5':
            example['Label'] = 'partial'
        elif label == '1':
            example['Label'] = 'correct'

        print("-" * 40)

    return dataset


dataset_path = "../data/dataset_with_completions.json"
with open(dataset_path, 'r') as f:
    dataset = json.load(f)

labeled_dataset = manual_review(dataset)

output_path = "../data/dataset_with_labels.json"
with open(output_path, 'w') as f:
    json.dump(labeled_dataset, f, indent=4)
