# Understanding CTC Decoding for Vietnamese ASR

This document explains the Connectionist Temporal Classification (CTC) decoding process and how different tokenization levels affect Vietnamese Automatic Speech Recognition (ASR).

## CTC Basics

Connectionist Temporal Classification (CTC) is a technique for sequence-to-sequence learning without requiring aligned data. It was introduced to solve the problem of aligning speech frames with text labels in ASR.

### Key Concepts

1. **Blank Token (`<blank>`)**: 
   - Represents "no output" at a particular timestep
   - Helps with alignment between input and output sequences
   - Allows the model to predict the same character multiple times

2. **CTC Path Collapse**:
   - Remove repeated tokens (unless separated by blank)
   - Remove blank tokens
   - Example: 
     - CTC output: `h_ee_l_l__o` (where `_` is blank)
     - After collapse: `hello`

3. **Probability Distribution**:
   - At each timestep, the model outputs a probability distribution over all tokens plus blank
   - The probability of a transcription is the sum of probabilities of all possible alignments

## Decoding Algorithms

### 1. Greedy Decoding

The simplest approach: take the most likely token at each timestep, then apply CTC collapse rules.

**Example (Character-level):**
```
Timesteps:   1    2    3    4    5    6    7    8    9
Predictions: x    i    _    n    _    _    c    h    à
After collapse: "xin chà"
```

**Pros**: Fast, simple implementation
**Cons**: Doesn't consider language constraints, prone to errors

### 2. Beam Search Decoding

Maintains multiple hypotheses (beams) and explores them to find the most likely sequence.

**Steps**:
1. Initialize with empty sequence
2. For each timestep:
   - Extend each beam with possible tokens
   - Apply CTC rules
   - Score with acoustic and language model
   - Keep top-K beams
3. Return the highest scoring beam

**Parameters**:
- `beam_size`: Number of hypotheses to maintain (e.g., 10)
- `alpha`: Language model weight (e.g., 0.5)
- `beta`: Length penalty (e.g., 1.0)

**Pros**: More accurate, incorporates language model
**Cons**: Slower, more complex

## Tokenization Levels for Vietnamese

### 1. Character-level

**Example**: "Xin chào" → ["X", "i", "n", " ", "c", "h", "à", "o"]

**Vocabulary size**: ~100-200 tokens (Vietnamese alphabet + diacritics + special characters)

**Characteristics**:
- Longest sequences (many tokens per word)
- Smallest vocabulary
- No OOV issues
- Requires strong language model for coherence
- Slowest decoding due to sequence length

### 2. Subword-level (BPE)

**Example**: "Xin chào" → ["Xi", "n", " ch", "ào"]

**Vocabulary size**: 5,000-10,000 tokens

**Characteristics**:
- Balances sequence length and vocabulary size
- Handles word variations and morphology
- May produce suboptimal segmentation for Vietnamese
- Good general-purpose choice

### 3. Syllable-level

**Example**: "Xin chào" → ["Xin", "chào"]

**Vocabulary size**: ~5,000-20,000 tokens (Vietnamese syllables)

**Characteristics**:
- Natural unit for Vietnamese (spaces separate syllables)
- Shorter sequences than characters
- Potential OOV issues with rare syllables
- Good fit for Vietnamese-specific ASR

### 4. Word-level

**Example**: "Xin chào" → ["Xin_chào"]

**Vocabulary size**: 20,000-50,000 tokens

**Characteristics**:
- Shortest sequences (fewest tokens)
- Largest vocabulary
- Significant OOV issues
- Requires word segmentation
- Fastest decoding due to short sequences
- OOV fallback to subwords or characters needed

## Vietnamese-Specific Considerations

1. **Tonal Language**: Vietnamese has 6 tones marked by diacritics
   - Character-level handles this naturally
   - Other levels need to preserve tone information

2. **Syllable Structure**: Vietnamese words consist of one or more syllables
   - Syllable-level tokenization is linguistically motivated
   - Word boundaries can be ambiguous

3. **Word Segmentation**: Unlike English, Vietnamese doesn't use spaces to separate words
   - Spaces separate syllables, not necessarily words
   - Word-level tokenization requires additional segmentation

## Example: Decoding Process

Let's walk through an example of decoding the Vietnamese phrase "Xin chào" (Hello) using different tokenization levels:

### Character-level

1. **Acoustic model output** (probabilities at each timestep):
   ```
   t1: [x=0.8, i=0.1, _=0.1, ...]
   t2: [i=0.7, n=0.2, _=0.1, ...]
   t3: [n=0.6, _=0.3, i=0.1, ...]
   t4: [_=0.9, n=0.05, c=0.05, ...]
   t5: [c=0.7, _=0.2, h=0.1, ...]
   t6: [h=0.8, à=0.1, _=0.1, ...]
   t7: [à=0.6, _=0.3, o=0.1, ...]
   t8: [o=0.7, _=0.2, à=0.1, ...]
   ```

2. **Greedy decoding**:
   - Take most likely token at each timestep: "xinchào"
   - After CTC collapse: "xin chào"

3. **Beam search with LM**:
   - Consider multiple paths and language model score
   - Final output: "xin chào"

### Syllable-level

1. **Acoustic model output** (probabilities at each timestep):
   ```
   t1: [xin=0.6, _=0.3, xi=0.1, ...]
   t2: [_=0.7, xin=0.2, chào=0.1, ...]
   t3: [chào=0.8, _=0.1, hào=0.1, ...]
   ```

2. **Greedy decoding**:
   - Take most likely token at each timestep: "xin_chào"
   - After CTC collapse: "xin chào"

3. **Beam search with LM**:
   - Consider multiple paths and language model score
   - Final output: "xin chào"

## Conclusion

The choice of tokenization level significantly impacts both the accuracy and speed of Vietnamese ASR systems. Character-level provides the best coverage but is slower, while word-level offers the fastest decoding but struggles with OOV words.

For most Vietnamese ASR applications, syllable-level tokenization provides a good balance, as it aligns with the natural structure of the language while maintaining reasonable vocabulary size and sequence length.
