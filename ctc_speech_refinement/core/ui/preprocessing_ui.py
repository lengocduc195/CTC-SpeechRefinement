"""
User interface for audio preprocessing options.

This module provides a GUI for configuring audio preprocessing options,
including normalization, silence removal, VAD, noise reduction, and frequency normalization.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import json

from ctc_speech_refinement.core.preprocessing.audio import preprocess_audio, batch_preprocess
from ctc_speech_refinement.core.utils.file_utils import get_audio_files

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Vietnamese translations
VI_TRANSLATIONS = {
    "Audio Preprocessing": "Tiền xử lý âm thanh",
    "Input File/Directory": "Tệp/Thư mục đầu vào",
    "Output Directory": "Thư mục đầu ra",
    "Browse": "Duyệt",
    "Amplitude Normalization": "Chuẩn hóa biên độ",
    "Silence Removal": "Loại bỏ khoảng lặng",
    "Voice Activity Detection (VAD)": "Voice Activity Detection (VAD)",
    "VAD Method": "Phương pháp VAD",
    "Energy-based": "Dựa trên năng lượng",
    "Zero-crossing rate": "Tỷ lệ giao điểm không",
    "Noise Reduction": "Khử nhiễu",
    "Noise Reduction Method": "Phương pháp khử nhiễu",
    "Spectral Subtraction": "Trừ phổ",
    "Wiener Filter": "Bộ lọc Wiener",
    "Median Filter": "Bộ lọc trung vị",
    "Noise Reduce Library": "Thư viện khử nhiễu",
    "Frequency Normalization": "Chuẩn hóa tần số",
    "Frequency Normalization Method": "Phương pháp chuẩn hóa tần số",
    "Bandpass Filter": "Bộ lọc thông dải",
    "Pre-emphasis": "Nhấn mạnh trước",
    "Spectral Equalization": "Cân bằng phổ",
    "Combined": "Kết hợp",
    "Process": "Xử lý",
    "Cancel": "Hủy",
    "Processing...": "Đang xử lý...",
    "Processing completed!": "Xử lý hoàn tất!",
    "Error": "Lỗi",
    "Please select input file or directory": "Vui lòng chọn tệp hoặc thư mục đầu vào",
    "Please select output directory": "Vui lòng chọn thư mục đầu ra",
    "Select File": "Chọn tệp",
    "Select Directory": "Chọn thư mục",
    "Audio Files": "Tệp âm thanh",
    "Save Configuration": "Lưu cấu hình",
    "Load Configuration": "Tải cấu hình",
    "Configuration Files": "Tệp cấu hình",
    "Configuration saved": "Đã lưu cấu hình",
    "Configuration loaded": "Đã tải cấu hình",
    "Preview": "Xem trước",
    "Original": "Gốc",
    "Processed": "Đã xử lý",
    "Time (s)": "Thời gian (s)",
    "Amplitude": "Biên độ",
    "Preview not available": "Không có bản xem trước",
    "Please select a single audio file for preview": "Vui lòng chọn một tệp âm thanh để xem trước"
}

class PreprocessingUI:
    """GUI for audio preprocessing options."""
    
    def __init__(self, root: tk.Tk, language: str = "vi"):
        """
        Initialize the preprocessing UI.
        
        Args:
            root: Tkinter root window.
            language: Language for the UI. Options: "en", "vi".
        """
        self.root = root
        self.language = language
        self.processing_thread = None
        self.is_processing = False
        
        # Set window title and size
        self.root.title(self._translate("Audio Preprocessing"))
        self.root.geometry("800x700")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create input/output section
        self.create_io_section()
        
        # Create preprocessing options section
        self.create_preprocessing_options()
        
        # Create preview section
        self.create_preview_section()
        
        # Create buttons section
        self.create_buttons_section()
    
    def _translate(self, text: str) -> str:
        """Translate text to the selected language."""
        if self.language == "vi" and text in VI_TRANSLATIONS:
            return VI_TRANSLATIONS[text]
        return text
    
    def create_io_section(self):
        """Create input/output file selection section."""
        io_frame = ttk.LabelFrame(self.main_frame, text=self._translate("Input/Output"), padding=10)
        io_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Input file/directory
        ttk.Label(io_frame, text=self._translate("Input File/Directory")).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.input_path_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.input_path_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(io_frame, text=self._translate("Browse"), command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)
        
        # Output directory
        ttk.Label(io_frame, text=self._translate("Output Directory")).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_path_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.output_path_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(io_frame, text=self._translate("Browse"), command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)
    
    def create_preprocessing_options(self):
        """Create preprocessing options section."""
        options_frame = ttk.LabelFrame(self.main_frame, text=self._translate("Preprocessing Options"), padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Amplitude normalization
        self.normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text=self._translate("Amplitude Normalization"), variable=self.normalize_var).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Silence removal
        self.remove_silence_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text=self._translate("Silence Removal"), variable=self.remove_silence_var).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        # VAD
        self.vad_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text=self._translate("Voice Activity Detection (VAD)"), variable=self.vad_var, command=self.toggle_vad).grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        # VAD method
        ttk.Label(options_frame, text=self._translate("VAD Method")).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        self.vad_method_var = tk.StringVar(value="energy")
        self.vad_method_combo = ttk.Combobox(options_frame, textvariable=self.vad_method_var, state="disabled", width=20)
        self.vad_method_combo["values"] = ["energy", "zcr"]
        self.vad_method_combo.grid(row=2, column=2, padx=5, pady=5)
        
        # Noise reduction
        self.noise_reduction_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text=self._translate("Noise Reduction"), variable=self.noise_reduction_var, command=self.toggle_noise_reduction).grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Noise reduction method
        ttk.Label(options_frame, text=self._translate("Noise Reduction Method")).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        self.noise_reduction_method_var = tk.StringVar(value="spectral_subtraction")
        self.noise_reduction_method_combo = ttk.Combobox(options_frame, textvariable=self.noise_reduction_method_var, state="disabled", width=20)
        self.noise_reduction_method_combo["values"] = ["spectral_subtraction", "wiener", "median", "noisereduce"]
        self.noise_reduction_method_combo.grid(row=3, column=2, padx=5, pady=5)
        
        # Frequency normalization
        self.frequency_normalization_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text=self._translate("Frequency Normalization"), variable=self.frequency_normalization_var, command=self.toggle_frequency_normalization).grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Frequency normalization method
        ttk.Label(options_frame, text=self._translate("Frequency Normalization Method")).grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        self.frequency_normalization_method_var = tk.StringVar(value="bandpass")
        self.frequency_normalization_method_combo = ttk.Combobox(options_frame, textvariable=self.frequency_normalization_method_var, state="disabled", width=20)
        self.frequency_normalization_method_combo["values"] = ["bandpass", "preemphasis", "equalize", "combined"]
        self.frequency_normalization_method_combo.grid(row=4, column=2, padx=5, pady=5)
    
    def create_preview_section(self):
        """Create preview section."""
        preview_frame = ttk.LabelFrame(self.main_frame, text=self._translate("Preview"), padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create figure for preview
        self.preview_figure = plt.Figure(figsize=(7, 4), dpi=100)
        self.preview_canvas = FigureCanvasTkAgg(self.preview_figure, preview_frame)
        self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Preview button
        ttk.Button(preview_frame, text=self._translate("Preview"), command=self.preview_processing).pack(pady=5)
    
    def create_buttons_section(self):
        """Create buttons section."""
        buttons_frame = ttk.Frame(self.main_frame, padding=10)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Save/load configuration buttons
        ttk.Button(buttons_frame, text=self._translate("Save Configuration"), command=self.save_configuration).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text=self._translate("Load Configuration"), command=self.load_configuration).pack(side=tk.LEFT, padx=5)
        
        # Process/cancel buttons
        self.process_button = ttk.Button(buttons_frame, text=self._translate("Process"), command=self.start_processing)
        self.process_button.pack(side=tk.RIGHT, padx=5)
        
        self.cancel_button = ttk.Button(buttons_frame, text=self._translate("Cancel"), command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.RIGHT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=15, pady=5)
        
        # Status label
        self.status_var = tk.StringVar()
        ttk.Label(self.main_frame, textvariable=self.status_var).pack(pady=5)
    
    def toggle_vad(self):
        """Toggle VAD method combobox state."""
        if self.vad_var.get():
            self.vad_method_combo["state"] = "readonly"
            # Disable silence removal if VAD is enabled
            self.remove_silence_var.set(False)
        else:
            self.vad_method_combo["state"] = "disabled"
    
    def toggle_noise_reduction(self):
        """Toggle noise reduction method combobox state."""
        if self.noise_reduction_var.get():
            self.noise_reduction_method_combo["state"] = "readonly"
        else:
            self.noise_reduction_method_combo["state"] = "disabled"
    
    def toggle_frequency_normalization(self):
        """Toggle frequency normalization method combobox state."""
        if self.frequency_normalization_var.get():
            self.frequency_normalization_method_combo["state"] = "readonly"
        else:
            self.frequency_normalization_method_combo["state"] = "disabled"
    
    def browse_input(self):
        """Browse for input file or directory."""
        file_path = filedialog.askopenfilename(
            title=self._translate("Select File"),
            filetypes=[(self._translate("Audio Files"), "*.wav *.mp3 *.flac *.ogg")]
        )
        if file_path:
            self.input_path_var.set(file_path)
    
    def browse_output(self):
        """Browse for output directory."""
        dir_path = filedialog.askdirectory(title=self._translate("Select Directory"))
        if dir_path:
            self.output_path_var.set(dir_path)
    
    def get_preprocessing_options(self) -> Dict[str, Any]:
        """Get preprocessing options from UI."""
        return {
            "normalize": self.normalize_var.get(),
            "remove_silence_flag": self.remove_silence_var.get(),
            "apply_vad_flag": self.vad_var.get(),
            "vad_method": self.vad_method_var.get(),
            "reduce_noise_flag": self.noise_reduction_var.get(),
            "noise_reduction_method": self.noise_reduction_method_var.get(),
            "normalize_frequency_flag": self.frequency_normalization_var.get(),
            "frequency_normalization_method": self.frequency_normalization_method_var.get()
        }
    
    def set_preprocessing_options(self, options: Dict[str, Any]):
        """Set preprocessing options in UI."""
        if "normalize" in options:
            self.normalize_var.set(options["normalize"])
        if "remove_silence_flag" in options:
            self.remove_silence_var.set(options["remove_silence_flag"])
        if "apply_vad_flag" in options:
            self.vad_var.set(options["apply_vad_flag"])
            self.toggle_vad()
        if "vad_method" in options:
            self.vad_method_var.set(options["vad_method"])
        if "reduce_noise_flag" in options:
            self.noise_reduction_var.set(options["reduce_noise_flag"])
            self.toggle_noise_reduction()
        if "noise_reduction_method" in options:
            self.noise_reduction_method_var.set(options["noise_reduction_method"])
        if "normalize_frequency_flag" in options:
            self.frequency_normalization_var.set(options["normalize_frequency_flag"])
            self.toggle_frequency_normalization()
        if "frequency_normalization_method" in options:
            self.frequency_normalization_method_var.set(options["frequency_normalization_method"])
    
    def save_configuration(self):
        """Save preprocessing configuration to file."""
        file_path = filedialog.asksaveasfilename(
            title=self._translate("Save Configuration"),
            filetypes=[(self._translate("Configuration Files"), "*.json")],
            defaultextension=".json"
        )
        if file_path:
            config = self.get_preprocessing_options()
            with open(file_path, "w") as f:
                json.dump(config, f, indent=4)
            messagebox.showinfo(
                self._translate("Configuration saved"),
                f"{self._translate('Configuration saved to')} {file_path}"
            )
    
    def load_configuration(self):
        """Load preprocessing configuration from file."""
        file_path = filedialog.askopenfilename(
            title=self._translate("Load Configuration"),
            filetypes=[(self._translate("Configuration Files"), "*.json")]
        )
        if file_path:
            try:
                with open(file_path, "r") as f:
                    config = json.load(f)
                self.set_preprocessing_options(config)
                messagebox.showinfo(
                    self._translate("Configuration loaded"),
                    f"{self._translate('Configuration loaded from')} {file_path}"
                )
            except Exception as e:
                messagebox.showerror(
                    self._translate("Error"),
                    f"{self._translate('Error loading configuration')}: {str(e)}"
                )
    
    def preview_processing(self):
        """Preview processing on the selected audio file."""
        input_path = self.input_path_var.get()
        if not input_path or not os.path.isfile(input_path):
            messagebox.showwarning(
                self._translate("Preview not available"),
                self._translate("Please select a single audio file for preview")
            )
            return
        
        # Clear figure
        self.preview_figure.clear()
        
        try:
            # Load original audio
            from ctc_speech_refinement.core.preprocessing.audio import load_audio
            original_audio, sample_rate = load_audio(input_path)
            
            # Process audio with selected options
            options = self.get_preprocessing_options()
            processed_audio, _ = preprocess_audio(
                input_path,
                **options
            )
            
            # Plot original and processed audio
            ax1 = self.preview_figure.add_subplot(211)
            ax1.plot(np.arange(len(original_audio)) / sample_rate, original_audio)
            ax1.set_title(self._translate("Original"))
            ax1.set_xlabel(self._translate("Time (s)"))
            ax1.set_ylabel(self._translate("Amplitude"))
            
            ax2 = self.preview_figure.add_subplot(212)
            ax2.plot(np.arange(len(processed_audio)) / sample_rate, processed_audio)
            ax2.set_title(self._translate("Processed"))
            ax2.set_xlabel(self._translate("Time (s)"))
            ax2.set_ylabel(self._translate("Amplitude"))
            
            self.preview_figure.tight_layout()
            self.preview_canvas.draw()
            
        except Exception as e:
            messagebox.showerror(
                self._translate("Error"),
                f"{self._translate('Error generating preview')}: {str(e)}"
            )
    
    def start_processing(self):
        """Start audio preprocessing."""
        input_path = self.input_path_var.get()
        output_path = self.output_path_var.get()
        
        if not input_path:
            messagebox.showwarning(
                self._translate("Error"),
                self._translate("Please select input file or directory")
            )
            return
        
        if not output_path:
            messagebox.showwarning(
                self._translate("Error"),
                self._translate("Please select output directory")
            )
            return
        
        # Disable UI during processing
        self.process_button["state"] = tk.DISABLED
        self.cancel_button["state"] = tk.NORMAL
        self.is_processing = True
        
        # Get preprocessing options
        options = self.get_preprocessing_options()
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self.process_audio,
            args=(input_path, output_path, options)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Update status
        self.status_var.set(self._translate("Processing..."))
        self.progress_var.set(0)
        
        # Check progress periodically
        self.root.after(100, self.check_progress)
    
    def process_audio(self, input_path: str, output_path: str, options: Dict[str, Any]):
        """Process audio files."""
        try:
            # Get list of audio files
            if os.path.isfile(input_path):
                file_paths = [input_path]
            else:
                file_paths = get_audio_files(input_path)
            
            # Process files
            batch_preprocess(
                file_paths,
                output_dir=output_path,
                **options
            )
            
            # Update status
            if self.is_processing:  # Check if not cancelled
                self.status_var.set(self._translate("Processing completed!"))
                self.progress_var.set(100)
        
        except Exception as e:
            # Handle errors
            logger.error(f"Error processing audio: {str(e)}")
            self.status_var.set(f"{self._translate('Error')}: {str(e)}")
        
        finally:
            # Re-enable UI
            self.is_processing = False
            self.process_button["state"] = tk.NORMAL
            self.cancel_button["state"] = tk.DISABLED
    
    def check_progress(self):
        """Check processing progress."""
        if self.is_processing:
            # Update progress (in a real app, you would calculate actual progress)
            # For now, just increment slightly
            current_progress = self.progress_var.get()
            if current_progress < 90:  # Cap at 90% until complete
                self.progress_var.set(current_progress + 1)
            
            # Check again after a delay
            self.root.after(100, self.check_progress)
    
    def cancel_processing(self):
        """Cancel audio preprocessing."""
        if self.is_processing:
            self.is_processing = False
            self.status_var.set(self._translate("Processing cancelled"))
            self.process_button["state"] = tk.NORMAL
            self.cancel_button["state"] = tk.DISABLED

def main():
    """Run the preprocessing UI."""
    root = tk.Tk()
    app = PreprocessingUI(root, language="vi")
    root.mainloop()

if __name__ == "__main__":
    main()
