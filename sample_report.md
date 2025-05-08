# Vietnamese ASR Tokenization Evaluation Report

## Summary

| Tokenizer | CER | WER | SER | Greedy Time (ms) | Beam Time (ms) |
| --- | --- | --- | --- | --- | --- |
| character | 0.0512 | 0.1245 | 0.1245 | 3.21 | 42.56 |
| subword | 0.0678 | 0.0987 | 0.0987 | 2.15 | 28.34 |
| syllable | 0.0823 | 0.0823 | 0.0823 | 1.43 | 18.92 |
| word | 0.1134 | 0.0756 | 0.0756 | 0.87 | 12.45 |

## Comparison Plots

### Error Rates by Tokenization Level
![Error Rates](sample_error_rates.png)

### Decoding Latency by Tokenization Level
![Latency](sample_latency.png)

### Accuracy vs. Speed Trade-off
![Trade-off](sample_tradeoff.png)

## Analysis and Recommendations

### Tokenization Level Comparison

- **Character-level:**
  - Pros: Smallest vocabulary, no OOV issues, handles all Vietnamese characters
  - Cons: Longer sequences, slower decoding, requires stronger language model
  - Best for: Maximum coverage of Vietnamese text, handling rare words

- **Subword-level (BPE):**
  - Pros: Good balance between vocabulary size and sequence length, handles word variations
  - Cons: May produce suboptimal segmentation for Vietnamese
  - Best for: General-purpose ASR with good balance of accuracy and speed

- **Syllable-level:**
  - Pros: Natural unit for Vietnamese, shorter sequences than characters
  - Cons: Larger vocabulary than characters, potential OOV issues
  - Best for: Vietnamese-specific ASR where syllables are well-defined

- **Word-level:**
  - Pros: Shortest sequences, fastest decoding, direct mapping to meaning
  - Cons: Largest vocabulary, significant OOV issues, requires word segmentation
  - Best for: Domain-specific applications with limited vocabulary, real-time requirements

### Recommendations

- **For real-time applications:** Consider syllable or word-level tokenization with greedy decoding
- **For maximum accuracy:** Use character or subword-level with beam search and a strong LM
- **For Vietnamese-specific applications:** Syllable-level tokenization offers a good balance
- **For general-purpose ASR:** Subword-level (BPE) with 5,000-10,000 tokens provides flexibility

## Sample Decoding Results

### Sample 1: "Xin chào, tôi đang thử nghiệm hệ thống nhận dạng tiếng nói."

| Tokenizer | Greedy Decoding | Beam Search Decoding | CER | WER |
| --- | --- | --- | --- | --- |
| character | Xin chào, tôi đang thử nghiệm hệ thống nhận dạng tiếng nói. | Xin chào, tôi đang thử nghiệm hệ thống nhận dạng tiếng nói. | 0.0000 | 0.0000 |
| subword | Xin chào, tôi đang thử nghiệm hệ thống nhận dạng tiếng nói. | Xin chào, tôi đang thử nghiệm hệ thống nhận dạng tiếng nói. | 0.0000 | 0.0000 |
| syllable | Xin chào tôi đang thử nghiệm hệ thống nhận dạng tiếng nói | Xin chào, tôi đang thử nghiệm hệ thống nhận dạng tiếng nói. | 0.0169 | 0.0000 |
| word | Xin chào tôi đang thử nghiệm hệ thống nhận dạng tiếng nói | Xin chào, tôi đang thử nghiệm hệ thống nhận dạng tiếng nói. | 0.0169 | 0.0000 |

### Sample 2: "Công nghệ nhận dạng tiếng nói đã phát triển rất nhanh trong những năm gần đây."

| Tokenizer | Greedy Decoding | Beam Search Decoding | CER | WER |
| --- | --- | --- | --- | --- |
| character | Công nghệ nhận dạng tiếng nói đã phát triển rất nhanh trong những năm gần đây. | Công nghệ nhận dạng tiếng nói đã phát triển rất nhanh trong những năm gần đây. | 0.0000 | 0.0000 |
| subword | Công nghệ nhận dạng tiếng nói đã phát triển rất nhanh trong năm gần đây. | Công nghệ nhận dạng tiếng nói đã phát triển rất nhanh trong những năm gần đây. | 0.0147 | 0.0909 |
| syllable | Công nghệ nhận dạng tiếng nói đã phát triển rất nhanh trong năm gần đây | Công nghệ nhận dạng tiếng nói đã phát triển rất nhanh trong những năm gần đây. | 0.0294 | 0.0909 |
| word | Công nghệ nhận dạng tiếng nói đã phát triển rất nhanh trong năm gần đây | Công nghệ nhận dạng tiếng nói đã phát triển rất nhanh trong những năm gần đây. | 0.0294 | 0.0909 |

## Conclusion

The choice of tokenization level for Vietnamese ASR involves a trade-off between accuracy and speed. Character-level tokenization provides the best character-level accuracy but is slower, while word-level tokenization offers the fastest decoding but may struggle with out-of-vocabulary words.

For most applications, subword or syllable-level tokenization provides a good balance between accuracy and speed. The use of beam search with a language model significantly improves accuracy across all tokenization levels, especially for character and subword tokenization.

When implementing a Vietnamese ASR system, consider the specific requirements of your application and choose the tokenization level accordingly.
