import json
import os
import yaml

# Write json
def save_json(content, save_path):
    with open(save_path, 'w') as file:
        json.dump(content, file, ensure_ascii=False, indent=4)

# Load json
def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        content = json.load(file)
        return content

# Load yml
def load_yml(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return None


    