"""
Script to fix null bytes in Python files.
"""

import os
import glob

def fix_null_bytes_in_file(file_path):
    """
    Fix null bytes in a Python file.
    
    Args:
        file_path: Path to the Python file.
    """
    print(f"Fixing null bytes in {file_path}")
    
    try:
        # Read the file in binary mode
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Replace null bytes with empty string
        content = content.replace(b'\x00', b'')
        
        # Write the file back
        with open(file_path, 'wb') as f:
            f.write(content)
            
        print(f"  Fixed {file_path}")
    except Exception as e:
        print(f"  Error fixing {file_path}: {str(e)}")

def fix_init_files(directory):
    """
    Fix __init__.py files in a directory and its subdirectories.
    
    Args:
        directory: Directory to search for __init__.py files.
    """
    print(f"Fixing __init__.py files in directory: {directory}")
    
    # Find all __init__.py files
    init_files = glob.glob(os.path.join(directory, '**', '__init__.py'), recursive=True)
    
    for file_path in init_files:
        # Create a new __init__.py file with proper content
        with open(file_path, 'w', encoding='utf-8') as f:
            module_name = os.path.basename(os.path.dirname(file_path))
            f.write(f'"""\n{module_name} module for CTC-SpeechRefinement.\n"""')
        print(f"  Created new {file_path}")

def fix_all_python_files(directory):
    """
    Fix null bytes in all Python files in a directory and its subdirectories.
    
    Args:
        directory: Directory to search for Python files.
    """
    print(f"Fixing null bytes in all Python files in directory: {directory}")
    
    # Find all Python files
    python_files = glob.glob(os.path.join(directory, '**', '*.py'), recursive=True)
    
    for file_path in python_files:
        fix_null_bytes_in_file(file_path)

if __name__ == "__main__":
    # Fix __init__.py files
    fix_init_files('ctc_speech_refinement')
    
    # Fix null bytes in all Python files
    fix_all_python_files('ctc_speech_refinement')
