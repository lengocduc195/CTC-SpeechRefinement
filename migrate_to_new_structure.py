"""
Script to migrate the existing code to the new directory structure.

This script copies files from the old structure to the new structure,
updating imports as needed.
"""

import os
import shutil
import re
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
OLD_ROOT = Path(".")
NEW_ROOT = Path("ctc_speech_refinement")

# Define mapping of old paths to new paths
PATH_MAPPING = {
    # Core modules
    "src/preprocessing": NEW_ROOT / "core/preprocessing",
    "src/features": NEW_ROOT / "core/features",
    "src/models": NEW_ROOT / "core/models",
    "src/decoder": NEW_ROOT / "core/decoder",
    "src/utils": NEW_ROOT / "core/utils",
    "src/eda": NEW_ROOT / "core/eda",
    
    # Apps
    "transcribe.py": NEW_ROOT / "apps/transcription/transcribe.py",
    "run_speculative_decoding.py": NEW_ROOT / "apps/speculative_decoding/run.py",
    "run_audio_eda.py": NEW_ROOT / "apps/audio_eda/run.py",
    "src/ui": NEW_ROOT / "apps/ui",
    
    # Config
    "config": NEW_ROOT / "config",
    
    # Tests
    "tests": NEW_ROOT / "tests",
    
    # Docs
    "docs": NEW_ROOT / "docs",
}

# Define import replacements
IMPORT_REPLACEMENTS = [
    (r"from src\.preprocessing", "from ctc_speech_refinement.core.preprocessing"),
    (r"from src\.features", "from ctc_speech_refinement.core.features"),
    (r"from src\.models", "from ctc_speech_refinement.core.models"),
    (r"from src\.decoder", "from ctc_speech_refinement.core.decoder"),
    (r"from src\.utils", "from ctc_speech_refinement.core.utils"),
    (r"from src\.eda", "from ctc_speech_refinement.core.eda"),
    (r"from src\.ui", "from ctc_speech_refinement.apps.ui"),
    (r"from config\.", "from ctc_speech_refinement.config."),
]

def update_imports(content):
    """Update import statements in the file content."""
    updated_content = content
    for old_pattern, new_pattern in IMPORT_REPLACEMENTS:
        updated_content = re.sub(old_pattern, new_pattern, updated_content)
    return updated_content

def copy_and_update_file(src_path, dest_path):
    """Copy a file and update its imports."""
    # Create destination directory if it doesn't exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Read source file
    with open(src_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Update imports
    updated_content = update_imports(content)
    
    # Write to destination file
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(updated_content)
    
    logger.info(f"Copied and updated: {src_path} -> {dest_path}")

def copy_directory(src_dir, dest_dir):
    """Copy a directory and update imports in Python files."""
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Copy files
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dest_path = os.path.join(dest_dir, item)
        
        if os.path.isdir(src_path):
            # Recursively copy subdirectory
            copy_directory(src_path, dest_path)
        elif src_path.endswith(".py"):
            # Copy and update Python file
            copy_and_update_file(src_path, dest_path)
        else:
            # Copy other files as is
            shutil.copy2(src_path, dest_path)
            logger.info(f"Copied: {src_path} -> {dest_path}")

def migrate_code():
    """Migrate code from old structure to new structure."""
    logger.info("Starting code migration...")
    
    # Process each mapping
    for old_path_str, new_path in PATH_MAPPING.items():
        old_path = OLD_ROOT / old_path_str
        
        if old_path.is_dir():
            # Copy directory
            copy_directory(old_path, new_path)
        elif old_path.is_file():
            # Copy file
            copy_and_update_file(old_path, new_path)
        else:
            logger.warning(f"Path not found: {old_path}")
    
    logger.info("Code migration completed!")

if __name__ == "__main__":
    migrate_code()
