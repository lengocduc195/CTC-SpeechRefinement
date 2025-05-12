# Nhận dạng tiếng nói thực sự cho tiếng Việt

Dự án này cung cấp các công cụ để thực hiện nhận dạng tiếng nói thực sự (ASR - Automatic Speech Recognition) cho tiếng Việt sử dụng các mô hình pre-trained hiện đại.

## Tính năng

- Hỗ trợ nhiều mô hình ASR pre-trained cho tiếng Việt:
  - Wav2Vec2 (nguyenvulebinh/wav2vec2-base-vietnamese-250h)
  - PhoWhisper (vinai/PhoWhisper-large)
- Xử lý batch các file âm thanh
- Đánh giá kết quả nhận dạng với các chỉ số WER (Word Error Rate) và CER (Character Error Rate)
- So sánh kết quả giữa phương pháp mô phỏng và phương pháp nhận dạng thực sự

## Cài đặt

### Yêu cầu

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.18+
- Các thư viện khác được liệt kê trong file requirements.txt

### Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

## Sử dụng

### Nhận dạng tiếng nói thực sự

Để thực hiện nhận dạng tiếng nói thực sự, sử dụng script `real_asr.py`:

```bash
python real_asr.py --input_dir data/audio --output_dir results/transcripts --model_type wav2vec2
```

#### Các tham số

- `--input_dir`: Thư mục chứa các file âm thanh cần nhận dạng
- `--output_dir`: Thư mục để lưu kết quả nhận dạng
- `--model_type`: Loại mô hình ASR (wav2vec2 hoặc whisper)
- `--model_size`: Kích thước mô hình (default, large, medium, small)
- `--custom_model_path`: Đường dẫn đến mô hình tùy chỉnh
- `--batch_size`: Kích thước batch cho xử lý
- `--device`: Thiết bị để sử dụng (cuda, cpu)
- `--reference_dir`: Thư mục chứa các transcript tham chiếu để đánh giá
- `--normalize_audio`: Chuẩn hóa âm thanh

### So sánh phương pháp mô phỏng và phương pháp nhận dạng thực sự

Để so sánh kết quả giữa phương pháp mô phỏng (sử dụng ctc_eval.py) và phương pháp nhận dạng thực sự (sử dụng real_asr.py), sử dụng script `compare_asr_methods.py`:

```bash
python compare_asr_methods.py --input_dir data/audio --reference_dir data/transcripts
```

#### Các tham số

- `--input_dir`: Thư mục chứa các file âm thanh
- `--reference_dir`: Thư mục chứa các transcript tham chiếu
- `--output_dir`: Thư mục để lưu kết quả so sánh
- `--mock_output_dir`: Thư mục để lưu kết quả nhận dạng mô phỏng
- `--real_output_dir`: Thư mục để lưu kết quả nhận dạng thực sự
- `--real_model_type`: Loại mô hình ASR thực sự
- `--real_model_size`: Kích thước mô hình ASR thực sự
- `--tokenizers`: Danh sách các tokenizer cho phương pháp mô phỏng
- `--beam_size`: Kích thước beam cho phương pháp mô phỏng

## Các mô hình được hỗ trợ

### Wav2Vec2

- `nguyenvulebinh/wav2vec2-base-vietnamese-250h`: Mô hình Wav2Vec2 base được pre-trained trên 13k giờ âm thanh tiếng Việt và fine-tuned trên 250 giờ dữ liệu có nhãn.
- `nguyenvulebinh/wav2vec2-large-vi-vlsp2020`: Phiên bản large của mô hình Wav2Vec2 cho tiếng Việt.

### PhoWhisper

- `vinai/PhoWhisper-large`: Mô hình PhoWhisper large cho tiếng Việt, được fine-tuned từ mô hình Whisper đa ngôn ngữ trên 844 giờ dữ liệu tiếng Việt.
- `vinai/PhoWhisper-medium`: Phiên bản medium của mô hình PhoWhisper.
- `vinai/PhoWhisper-small`: Phiên bản small của mô hình PhoWhisper.

## Ví dụ

### Nhận dạng tiếng nói với Wav2Vec2

```bash
python real_asr.py --input_dir data/audio --output_dir results/wav2vec2_transcripts --model_type wav2vec2
```

### Nhận dạng tiếng nói với PhoWhisper

```bash
python real_asr.py --input_dir data/audio --output_dir results/whisper_transcripts --model_type whisper
```

### So sánh Wav2Vec2 và PhoWhisper

```bash
# Chạy Wav2Vec2
python real_asr.py --input_dir data/audio --output_dir results/wav2vec2_transcripts --model_type wav2vec2

# Chạy PhoWhisper
python real_asr.py --input_dir data/audio --output_dir results/whisper_transcripts --model_type whisper

# So sánh kết quả
python compare_results.py --reference_dir data/transcripts --wav2vec2_dir results/wav2vec2_transcripts --whisper_dir results/whisper_transcripts --output_dir results/comparison
```

## Đánh giá kết quả

Kết quả đánh giá được lưu trong thư mục output và bao gồm:

- WER (Word Error Rate): Tỷ lệ lỗi từ
- CER (Character Error Rate): Tỷ lệ lỗi ký tự

Các biểu đồ so sánh cũng được tạo ra để trực quan hóa hiệu suất của các phương pháp khác nhau.

## Tài liệu tham khảo

- [Wav2Vec2 Vietnamese](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h)
- [PhoWhisper](https://huggingface.co/vinai/PhoWhisper-large)
- [Transformers Documentation](https://huggingface.co/docs/transformers/index)

## Giấy phép

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

## Liên hệ

Nếu bạn có bất kỳ câu hỏi hoặc đề xuất nào, vui lòng tạo một issue trong repository này.
