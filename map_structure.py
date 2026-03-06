# File: map_structure.py
import os
import json

def map_directory_structure(root_path):
    structure = {}
    
    for root, dirs, files in os.walk(root_path):
        # Skip unwanted directories
        dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git', 'node_modules', '__MACOSX']]
        
        rel_path = os.path.relpath(root, root_path)
        if rel_path == '.':
            rel_path = 'root'
        
        structure[rel_path] = {
            'folders': dirs,
            'files': files
        }
    
    return structure

if __name__ == "__main__":
    structure = map_directory_structure('.')
    
    with open('project_structure.json', 'w') as f:
        json.dump(structure, f, indent=2)
    
    print("✅ Project structure saved to project_structure.json")
    print(f"📁 Total folders mapped: {len(structure)}")
    print(f"📄 Total files: {sum(len(v['files']) for v in structure.values())}")