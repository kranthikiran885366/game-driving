import os
import re
from pathlib import Path

def get_module_path(file_path):
    """Get the module path for a file relative to the project root."""
    parts = Path(file_path).parts
    try:
        src_idx = parts.index('src')
        module_parts = parts[src_idx:]
        return '.'.join(module_parts).replace('.py', '')
    except ValueError:
        return None

def fix_imports_in_file(file_path):
    """Fix imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Get all Python files in the project
        python_files = []
        for root, _, files in os.walk('.'):
            if 'venv' in root or '__pycache__' in root:
                continue
            for f in files:
                if f.endswith('.py') and f != 'fix_all_imports.py':
                    full_path = os.path.join(root, f)
                    python_files.append(full_path)
        
        # Create a mapping of module names to their full paths
        module_map = {}
        for f in python_files:
            module_name = os.path.basename(f).replace('.py', '')
            module_map[module_name] = get_module_path(f)
        
        # Fix imports
        for module_name, full_path in module_map.items():
            if not full_path:
                continue
                
            # Handle 'from x import' pattern
            pattern = fr'(from\s+)(["\']?)(\.*\b{re.escape(module_name)}\b["\']?)(\s+import)'
            replacement = fr'\1\2{full_path}\4'
            content = re.sub(pattern, replacement, content)
            
            # Handle 'import x' pattern (only if not already a full path)
            if not any(c in module_name for c in ['./', '../']):
                pattern = fr'(import\s+)(["\']?)(\b{re.escape(module_name)}\b)(?!\.)'
                replacement = fr'\1\2{full_path.split(".")[-1]}'
                content = re.sub(pattern, replacement, content)
        
        # Fix relative imports within the same package
        if file_path.startswith('src/'):
            rel_depth = len(Path(file_path).parts) - 2  # -2 for filename and src/
            parent_ref = '.'.join(['..'] * rel_depth)
            content = content.replace('from .', f'from {parent_ref}.')
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error fixing imports in {file_path}: {e}")
        return False

def main():
    print("Fixing imports in all Python files...")
    
    # Get all Python files
    python_files = []
    for root, _, files in os.walk('.'):
        if 'venv' in root or '__pycache__' in root or '.git' in root:
            continue
        for file in files:
            if file.endswith('.py') and file != 'fix_all_imports.py':
                python_files.append(os.path.join(root, file))
    
    # Process all files
    updated_count = 0
    for file_path in python_files:
        if fix_imports_in_file(file_path):
            print(f"Fixed imports in: {file_path}")
            updated_count += 1
    
    print(f"\nImport fixing complete! Processed {len(python_files)} files, updated {updated_count} files.")
    print("\nNext steps:")
    print("1. Run 'python test_imports.py' to verify all imports work")
    print("2. Run 'python -m pytest' to run tests")
    print("3. Run 'python src/core/driving_hand.py' to test the game")

if __name__ == "__main__":
    main()
