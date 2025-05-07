"""
Script to restructure the project.

This script performs the following steps:
1. Creates the new directory structure
2. Migrates the code to the new structure
3. Updates the README.md file
"""

import os
import shutil
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command):
    """Run a shell command."""
    logger.info(f"Running command: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {result.stderr}")
    else:
        logger.info(f"Command succeeded: {result.stdout}")
    return result

def restructure_project():
    """Restructure the project."""
    logger.info("Starting project restructuring...")
    
    # 1. Run the migration script
    logger.info("Migrating code to new structure...")
    run_command("python migrate_to_new_structure.py")
    
    # 2. Update the README.md file
    logger.info("Updating README.md...")
    shutil.copy("README_NEW.md", "README.md")
    
    # 3. Create symbolic links for convenience
    logger.info("Creating symbolic links...")
    
    # On Windows, we need to use different commands for symbolic links
    if os.name == 'nt':  # Windows
        run_command("mklink run_transcription.bat run_transcription.py")
        run_command("mklink run_speculative_decoding.bat run_speculative_decoding.py")
        run_command("mklink run_audio_eda.bat run_audio_eda_new.py")
        run_command("mklink run_preprocessing_ui.bat run_preprocessing_ui_new.py")
    else:  # Unix-like
        run_command("ln -sf run_transcription.py run_transcription")
        run_command("ln -sf run_speculative_decoding.py run_speculative_decoding")
        run_command("ln -sf run_audio_eda_new.py run_audio_eda")
        run_command("ln -sf run_preprocessing_ui_new.py run_preprocessing_ui")
        run_command("chmod +x run_transcription run_speculative_decoding run_audio_eda run_preprocessing_ui")
    
    logger.info("Project restructuring completed!")
    logger.info("Please install the package in development mode: pip install -e ctc_speech_refinement")

if __name__ == "__main__":
    restructure_project()
