import os
import numpy as np
import torch
import librosa
import json
from jiwer import wer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder

def preprocess_audio(file_path):
    """Tiền xử lý file âm thanh thành dạng phù hợp với mô hình."""
    try:
        # Kiểm tra file tồn tại
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file âm thanh: {file_path}")
        
        # Đọc và chuyển đổi về tần số lấy mẫu 16kHz
        audio, orig_sr = librosa.load(file_path, sr=None)
        if orig_sr != 16000:
            # print(f"Chuyển đổi từ {orig_sr}Hz sang 16kHz...")
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=16000)
            
        # # Cắt nếu độ dài quá 10 giây
        # if len(audio) > 160000:  # 10 giây ở 16kHz
        #     print("Cảnh báo: Âm thanh dài hơn 10 giây. Đang cắt...")
        #     audio = audio[:160000]
            
        return audio
        
    except Exception as e:
        raise RuntimeError(f"Lỗi khi xử lý file âm thanh: {e}")

def transcribe_audio(file_path):
    """Chuyển đổi âm thanh thành văn bản."""
    try:
        # Tiền xử lý âm thanh
        audio = preprocess_audio(file_path)
        
        # Tokenize với sampling_rate được chỉ định rõ ràng
        input_values = processor(audio, return_tensors="pt", padding="longest", sampling_rate=16000).input_values
        
        # Lấy logits từ mô hình
        with torch.no_grad():
            logits = model(input_values).logits.cpu().numpy()[0]
        
        # Giải mã
        if use_lm:
            # Sử dụng beam search với language model
            beam_results = decoder.decode_beams(logits, beam_width=100)
            transcription = beam_results[0][0]
        else:
            # Sử dụng argmax decoding cơ bản
            predicted_ids = torch.argmax(torch.tensor(logits), dim=-1)
            transcription = processor.decode(predicted_ids)
        
        return transcription
        
    except Exception as e:
        print(f"Lỗi khi chuyển đổi âm thanh: {e}")
        return None

def convert(file_path):
    """Chuyển đổi một file âm thanh và in kết quả."""
    result = transcribe_audio(file_path)
    if result:
        print(f"\nKết quả chuyển đổi ({os.path.basename(file_path)}):")
        print(result)
    return result

def evaluate_wer(reference_text, hypothesis_text):
    """Tính toán Word Error Rate giữa văn bản tham chiếu và văn bản nhận dạng."""
    if not reference_text or not hypothesis_text:
        return 1.0  # Trả về lỗi 100% nếu một trong hai chuỗi trống
    
    # Tính toán WER
    error_rate = wer(reference_text, hypothesis_text)
    return error_rate

def read_reference(reference_file):
    """Đọc văn bản tham chiếu từ file."""
    try:
        with open(reference_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Không thể đọc file tham chiếu {reference_file}: {e}")
        return None

def convert_list(audio_dir, reference_dir):
    """
    Xử lý tất cả các file âm thanh trong thư mục và đánh giá với file tham chiếu.
    
    Parameters:
    audio_dir (str): Đường dẫn đến thư mục chứa file âm thanh
    reference_dir (str): Đường dẫn đến thư mục chứa file tham chiếu
    
    Returns:
    dict: Kết quả theo định dạng {file_name: {transcription: str, reference: str, wer: float}}
    """
    results = {}
    total_wer = 0
    count = 0
    
    # Kiểm tra thư mục tồn tại
    if not os.path.isdir(audio_dir):
        print(f"Thư mục âm thanh không tồn tại: {audio_dir}")
        return results
    
    if not os.path.isdir(reference_dir):
        print(f"Thư mục tham chiếu không tồn tại: {reference_dir}")
        return results
    
    # Định dạng file âm thanh phổ biến
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']
    
    # Lấy danh sách file âm thanh
    audio_files = []
    for file_name in os.listdir(audio_dir):
        _, ext = os.path.splitext(file_name)
        if ext.lower() in audio_extensions:
            audio_files.append(file_name)
    
    print(f"Tìm thấy {len(audio_files)} file âm thanh. Đang bắt đầu xử lý...")
    
    # Xử lý từng file
    for file_name in audio_files:
        # Đường dẫn đầy đủ đến file âm thanh
        audio_path = os.path.join(audio_dir, file_name)
        
        # Tìm file tham chiếu tương ứng (giả sử có cùng tên file, đuôi .txt)
        base_name = os.path.splitext(file_name)[0]
        reference_path = os.path.join(reference_dir, f"{base_name}.txt")
        
        # Kiểm tra xem file tham chiếu có tồn tại không
        if not os.path.exists(reference_path):
            print(f"Không tìm thấy file tham chiếu {reference_path} cho {file_name}")
            continue
        
        # Chuyển đổi âm thanh thành văn bản
        transcription = transcribe_audio(audio_path)
        if not transcription:
            print(f"Không thể chuyển đổi file {file_name}")
            continue
        
        # Đọc văn bản tham chiếu
        reference_text = read_reference(reference_path)
        if not reference_text:
            continue
        
        # Tính toán WER
        error_rate = evaluate_wer(reference_text, transcription)
        total_wer += error_rate
        count += 1
        
        # Lưu kết quả
        results[file_name] = {
            "transcription": transcription,
            "reference": reference_text,
            "wer": error_rate
        }
        
        print(f"File: {file_name}")
        print(f"Tham chiếu: {reference_text}")
        print(f"Nhận dạng: {transcription}")
        print(f"WER: {error_rate:.4f} ({error_rate*100:.2f}%)")
        print("-" * 50)
    
    # Tính WER trung bình
    if count > 0:
        avg_wer = total_wer / count
        print(f"WER trung bình: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    
    # Lưu kết quả vào file JSON
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Đã lưu kết quả vào file evaluation_results.json")
    
    return results

# Mã sử dụng (ví dụ)
if __name__ == "__main__":
    # Tải mô hình và processor (giả sử đã được định nghĩa trước đó)
    model_name = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    
    # Thiết lập CTC decoder
    vocab_dict = processor.tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab = [x[1] for x in sort_vocab]
    
    # Kiểm tra mô hình ngôn ngữ
    lm_file = "vi_lm_4grams.bin"
    use_lm = False
    
    if os.path.exists(lm_file):
        try:
            import kenlm
            decoder = build_ctcdecoder(
                vocab,
                kenlm_model_path=lm_file,
                alpha=0.5,
                beta=1.0,
            )
            use_lm = True
            print("Đã tải thành công mô hình ngôn ngữ!")
        except Exception as e:
            print(f"Không thể tải mô hình ngôn ngữ: {e}")
            decoder = build_ctcdecoder(vocab)
    else:
        print("Không tìm thấy mô hình ngôn ngữ, sử dụng giải mã cơ bản")
        decoder = build_ctcdecoder(vocab)
    
    # Sử dụng các hàm
    while True:
        print("\nChọn chế độ:")
        print("1. Xử lý một file âm thanh")
        print("2. Xử lý thư mục và đánh giá")
        print("3. Mặc định (Xử lý thư mục data/audio và đánh giá với data/transcripts)")
        print("4. Mặc định phát hiện lỗi (Xử lý thư mục data/errordetect và đánh giá với data/transcripts)")
        print("q. Thoát")
        
        choice = input("Lựa chọn của bạn: ")
        
        if choice.lower() == 'q':
            break
            
        elif choice == '1':
            file_path = input("Nhập đường dẫn file âm thanh: ")
            convert(file_path)
            
        elif choice == '2':
            audio_dir = input("Nhập đường dẫn thư mục âm thanh: ")
            reference_dir = input("Nhập đường dẫn thư mục tham chiếu: ")
            convert_list(audio_dir, reference_dir)
        elif choice == '3':
            audio_dir = "data/audio"
            reference_dir = "data/transcripts"
            convert_list(audio_dir, reference_dir)
        elif choice == '4':
            audio_dir = "data/errordetect"
            reference_dir = "data/transcripts"
            convert_list(audio_dir, reference_dir)
            
        else:
            print("Lựa chọn không hợp lệ!")
