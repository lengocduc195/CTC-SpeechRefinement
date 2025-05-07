"""
Test script to check if imports work.
"""

import sys
sys.path.append('..')

try:
    from ctc_speech_refinement.core.preprocessing.audio import load_audio
    print("Import successful!")
except Exception as e:
    print(f"Import failed: {str(e)}")
