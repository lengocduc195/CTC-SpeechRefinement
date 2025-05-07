# Integration Guide

This guide explains how to integrate the CTC Speech Transcription system with other projects and systems.

## Integration as a Python Module

### Basic Integration

You can import and use the components of the system in your Python code:

```python
from src.preprocessing.audio import preprocess_audio
from src.models.acoustic_model import AcousticModel
from src.decoder.ctc_decoder import CTCDecoder

def transcribe_audio(audio_file):
    # Preprocess audio
    audio_data, sample_rate = preprocess_audio(audio_file)
    
    # Initialize model and decoder
    model = AcousticModel()
    decoder = CTCDecoder(model.processor)
    
    # Get logits from model
    logits = model.get_logits(audio_data, sample_rate)
    
    # Decode to get transcription
    transcription = decoder.decode(logits)
    
    return transcription
```

### Integration with Web Applications

Here's an example of how to integrate the system with a Flask web application:

```python
from flask import Flask, request, jsonify
import os
from src.preprocessing.audio import preprocess_audio
from src.models.acoustic_model import AcousticModel
from src.decoder.ctc_decoder import CTCDecoder

app = Flask(__name__)

# Initialize model and decoder (do this at startup to avoid reloading for each request)
model = AcousticModel()
decoder = CTCDecoder(model.processor)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the file temporarily
    temp_path = 'temp_audio.wav'
    file.save(temp_path)
    
    try:
        # Preprocess audio
        audio_data, sample_rate = preprocess_audio(temp_path)
        
        # Get logits from model
        logits = model.get_logits(audio_data, sample_rate)
        
        # Decode to get transcription
        transcription = decoder.decode(logits)
        
        return jsonify({'transcription': transcription})
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(debug=True)
```

## Integration with Command-Line Tools

### Using as a Command-Line Tool

The system can be used as a command-line tool in shell scripts:

```bash
#!/bin/bash

# Process all WAV files in a directory
for file in /path/to/audio/*.wav; do
    echo "Transcribing $file..."
    python transcribe.py --input_dir $(dirname "$file") --output_dir transcripts
done

# Process the results
echo "Processing transcriptions..."
cat transcripts/*.txt > all_transcriptions.txt
```

### Integration with Data Processing Pipelines

You can integrate the system into data processing pipelines:

```bash
#!/bin/bash

# Record audio
arecord -d 60 -f S16_LE -r 16000 recording.wav

# Transcribe audio
python transcribe.py --input_dir . --output_dir transcripts

# Process transcription
cat transcripts/recording.txt | grep -i "important" > important_parts.txt
```

## Integration with Other Programming Languages

### Using via Command-Line Interface

From other programming languages, you can call the system via its command-line interface:

#### Java Example:
```java
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class TranscriptionExample {
    public static String transcribeAudio(String audioFilePath) throws Exception {
        ProcessBuilder processBuilder = new ProcessBuilder(
            "python", "transcribe.py", 
            "--input_dir", new File(audioFilePath).getParent(),
            "--output_dir", "transcripts"
        );
        
        Process process = processBuilder.start();
        int exitCode = process.waitFor();
        
        if (exitCode != 0) {
            throw new Exception("Transcription failed with exit code " + exitCode);
        }
        
        // Read the transcription from the output file
        String transcriptionFile = "transcripts/" + new File(audioFilePath).getName().replace(".wav", ".txt");
        BufferedReader reader = new BufferedReader(new FileReader(transcriptionFile));
        StringBuilder transcription = new StringBuilder();
        String line;
        
        while ((line = reader.readLine()) != null) {
            transcription.append(line).append("\n");
        }
        
        reader.close();
        return transcription.toString();
    }
}
```

#### Node.js Example:
```javascript
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');

function transcribeAudio(audioFilePath) {
    return new Promise((resolve, reject) => {
        const inputDir = path.dirname(audioFilePath);
        const outputDir = 'transcripts';
        
        exec(`python transcribe.py --input_dir "${inputDir}" --output_dir "${outputDir}"`, (error, stdout, stderr) => {
            if (error) {
                reject(error);
                return;
            }
            
            const baseName = path.basename(audioFilePath, path.extname(audioFilePath));
            const transcriptionFile = path.join(outputDir, `${baseName}.txt`);
            
            fs.readFile(transcriptionFile, 'utf8', (err, data) => {
                if (err) {
                    reject(err);
                    return;
                }
                
                resolve(data);
            });
        });
    });
}

// Usage
transcribeAudio('/path/to/audio.wav')
    .then(transcription => console.log(transcription))
    .catch(error => console.error(error));
```

### Using via REST API

You can also create a REST API wrapper around the system:

1. Create a simple Flask API (as shown in the web application example)
2. Deploy it as a service
3. Call the API from any programming language

## Integration with Cloud Services

### Deploying as a Microservice

You can deploy the system as a microservice in a container:

1. Create a Dockerfile:
```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "api.py"]
```

2. Build and run the container:
```bash
docker build -t ctc-transcription .
docker run -p 5000:5000 ctc-transcription
```

3. Call the API:
```bash
curl -X POST -F "file=@audio.wav" http://localhost:5000/transcribe
```

### Integration with AWS Lambda

You can deploy the system as an AWS Lambda function:

1. Create a Lambda function with the necessary dependencies
2. Use S3 triggers to automatically transcribe uploaded audio files
3. Store the transcriptions back in S3 or in a database

## Extension Points

The system provides several extension points for integration:

1. **Custom Preprocessing**: You can add custom preprocessing steps for specific audio sources
2. **Custom Models**: You can integrate different acoustic models
3. **Custom Decoders**: You can implement custom decoding algorithms
4. **Custom Output Formats**: You can modify the output format to match your requirements
