# Hướng dẫn Tiền xử lý Âm thanh

Tài liệu này giải thích các tính năng tiền xử lý âm thanh có sẵn trong dự án CTC-SpeechRefinement.

## Tổng quan

Module tiền xử lý âm thanh cung cấp nhiều kỹ thuật để cải thiện chất lượng dữ liệu âm thanh trước khi nhận dạng giọng nói. Các kỹ thuật này bao gồm:

1. **Chuẩn hóa biên độ**: Chuẩn hóa biên độ của tín hiệu âm thanh.
2. **Loại bỏ khoảng lặng**: Loại bỏ các đoạn im lặng khỏi âm thanh.
3. **Voice Activity Detection (VAD)**: Phát hiện và trích xuất các đoạn giọng nói.
4. **Khử nhiễu**: Giảm nhiễu nền bằng nhiều phương pháp khác nhau.
5. **Chuẩn hóa tần số**: Chuẩn hóa nội dung tần số của âm thanh.

## Sử dụng giao diện người dùng

Giao diện người dùng tiền xử lý cung cấp một giao diện đồ họa để cấu hình và áp dụng các tùy chọn tiền xử lý cho các tệp âm thanh.

### Chạy giao diện người dùng

Để chạy giao diện người dùng tiền xử lý, sử dụng lệnh sau:

```bash
python run_preprocessing_ui_new.py --language vi
```

Tùy chọn:
- `--language` hoặc `-l`: Ngôn ngữ giao diện người dùng (en hoặc vi). Mặc định là "vi" (tiếng Việt).

### Tính năng giao diện người dùng

Giao diện người dùng bao gồm các phần sau:

1. **Đầu vào/Đầu ra**: Chọn tệp âm thanh đầu vào và thư mục đầu ra.
2. **Tùy chọn tiền xử lý**: Cấu hình các tùy chọn tiền xử lý.
3. **Xem trước**: Xem trước hiệu ứng của tiền xử lý trên tệp âm thanh đã chọn.
4. **Cấu hình**: Lưu và tải cấu hình tiền xử lý.

## Tùy chọn tiền xử lý

### Chuẩn hóa biên độ

Chuẩn hóa dữ liệu âm thanh để có giá trị trung bình bằng không và phương sai đơn vị, giúp chuẩn hóa mức âm lượng trên các bản ghi khác nhau.

### Loại bỏ khoảng lặng

Loại bỏ các đoạn im lặng khỏi âm thanh, có thể cải thiện độ chính xác nhận dạng giọng nói và giảm thời gian xử lý.

### Voice Activity Detection (VAD)

Phát hiện và trích xuất các đoạn giọng nói từ âm thanh, bỏ qua các đoạn không phải giọng nói.

Các phương pháp có sẵn:
- **Dựa trên năng lượng**: Phát hiện giọng nói dựa trên mức năng lượng của âm thanh.
- **Tỷ lệ giao điểm không**: Phát hiện giọng nói dựa trên tỷ lệ giao điểm không và mức năng lượng.

### Khử nhiễu

Giảm nhiễu nền trong âm thanh, có thể cải thiện độ chính xác nhận dạng giọng nói.

Các phương pháp có sẵn:
- **Trừ phổ**: Ước tính phổ nhiễu và trừ nó khỏi phổ âm thanh.
- **Bộ lọc Wiener**: Áp dụng bộ lọc Wiener để giảm nhiễu.
- **Bộ lọc trung vị**: Áp dụng bộ lọc trung vị để giảm nhiễu xung.
- **Thư viện khử nhiễu**: Sử dụng thư viện noisereduce để khử nhiễu.

### Chuẩn hóa tần số

Chuẩn hóa nội dung tần số của âm thanh, có thể cải thiện độ chính xác nhận dạng giọng nói.

Các phương pháp có sẵn:
- **Bộ lọc thông dải**: Áp dụng bộ lọc thông dải để tập trung vào dải tần số giọng nói.
- **Nhấn mạnh trước**: Tăng cường tần số cao để nhấn mạnh giọng nói.
- **Cân bằng phổ**: Làm phẳng phổ tần số.
- **Kết hợp**: Áp dụng kết hợp các phương pháp trên.

## Sử dụng API

Bạn cũng có thể sử dụng API tiền xử lý trực tiếp trong mã Python của bạn:

```python
from ctc_speech_refinement.core.preprocessing.audio import preprocess_audio, batch_preprocess

# Tiền xử lý một tệp âm thanh
audio_data, sample_rate = preprocess_audio(
    "data/test1/test1_01.wav",
    normalize=True,
    remove_silence_flag=True,
    apply_vad_flag=True,
    vad_method="energy",
    reduce_noise_flag=True,
    noise_reduction_method="spectral_subtraction",
    normalize_frequency_flag=True,
    frequency_normalization_method="bandpass"
)

# Tiền xử lý nhiều tệp âm thanh
results = batch_preprocess(
    ["data/test1/test1_01.wav", "data/test1/test1_02.wav"],
    output_dir="data/preprocessed",
    normalize=True,
    remove_silence_flag=True,
    apply_vad_flag=True,
    vad_method="energy",
    reduce_noise_flag=True,
    noise_reduction_method="spectral_subtraction",
    normalize_frequency_flag=True,
    frequency_normalization_method="bandpass"
)
```

## Sử dụng nâng cao

### Voice Activity Detection (VAD)

Bạn có thể sử dụng module VAD trực tiếp:

```python
from ctc_speech_refinement.core.preprocessing.vad import apply_vad, energy_vad, zcr_vad
import librosa

# Tải âm thanh
audio_data, sample_rate = librosa.load("data/test1/test1_01.wav", sr=16000)

# Áp dụng VAD
speech_audio = apply_vad(audio_data, sample_rate, method="energy")

# Lấy các vùng giọng nói
speech_regions = energy_vad(audio_data, sample_rate)
for start_time, end_time in speech_regions:
    print(f"Giọng nói từ {start_time:.2f}s đến {end_time:.2f}s")
```

### Khử nhiễu

Bạn có thể sử dụng module khử nhiễu trực tiếp:

```python
from ctc_speech_refinement.core.preprocessing.noise_reduction import reduce_noise, spectral_subtraction, wiener_filter
import librosa

# Tải âm thanh
audio_data, sample_rate = librosa.load("data/test1/test1_01.wav", sr=16000)

# Áp dụng khử nhiễu
denoised_audio = reduce_noise(audio_data, sample_rate, method="spectral_subtraction")

# Sử dụng phương pháp cụ thể
denoised_audio = spectral_subtraction(audio_data, sample_rate)
```

### Chuẩn hóa tần số

Bạn có thể sử dụng module chuẩn hóa tần số trực tiếp:

```python
from ctc_speech_refinement.core.preprocessing.frequency_normalization import normalize_frequency, apply_bandpass_filter, apply_preemphasis
import librosa

# Tải âm thanh
audio_data, sample_rate = librosa.load("data/test1/test1_01.wav", sr=16000)

# Áp dụng chuẩn hóa tần số
normalized_audio = normalize_frequency(audio_data, sample_rate, method="bandpass")

# Sử dụng phương pháp cụ thể
filtered_audio = apply_bandpass_filter(audio_data, sample_rate, low_freq=80.0, high_freq=8000.0)
```

## Thực hành tốt nhất

Để có kết quả nhận dạng giọng nói tối ưu, hãy xem xét quy trình tiền xử lý sau:

1. **Chuẩn hóa tần số**: Áp dụng lọc thông dải để tập trung vào dải tần số giọng nói (80-8000 Hz).
2. **Khử nhiễu**: Áp dụng trừ phổ hoặc lọc Wiener để giảm nhiễu nền.
3. **Voice Activity Detection**: Áp dụng VAD để trích xuất các đoạn giọng nói.
4. **Chuẩn hóa biên độ**: Chuẩn hóa biên độ của các đoạn giọng nói.

Quy trình này giúp cải thiện chất lượng dữ liệu âm thanh và có thể cải thiện đáng kể độ chính xác nhận dạng giọng nói.

## Xử lý sự cố

### Vấn đề thường gặp

1. **Không phát hiện giọng nói**: Thử điều chỉnh các tham số VAD hoặc sử dụng phương pháp VAD khác.
2. **Khử nhiễu quá mức**: Thử phương pháp khử nhiễu khác hoặc điều chỉnh các tham số.
3. **Âm thanh bị biến dạng**: Kiểm tra xem các tham số chuẩn hóa tần số có phù hợp với âm thanh của bạn không.

### Nhận trợ giúp

Nếu bạn gặp bất kỳ vấn đề nào với module tiền xử lý, vui lòng kiểm tra nhật ký để biết thông báo lỗi và tham khảo tài liệu API để biết thêm thông tin.
