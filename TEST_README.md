# Interview API Test Scripts

This directory contains test scripts to validate the Interview Chatbot API endpoints.

## Test Scripts

### 1. `test_interview_endpoints.py` - Full Integration Test
**Complete end-to-end test with real audio processing**

- Uses OpenAI TTS to generate candidate speech from text
- Tests actual audio transcription (STT) 
- Simulates a full interview session
- Saves audio samples for manual verification
- Tests all endpoints: `/start-interview`, `/answer-question`, `/get-evaluation`

**Requirements:**
- OpenAI API key in `.env` file
- Gemini API key in `.env` file
- Server running on `localhost:8000`

**Usage:**
```bash
python test_interview_endpoints.py
```

### 2. `simple_api_test.py` - API Logic Test  
**Fast test focusing on API endpoints without audio processing**

- Uses mock audio data (minimal valid WAV)
- Tests API request/response logic
- Faster execution (no TTS/STT processing)
- Good for debugging API issues

**Requirements:**
- Server running on `localhost:8000`
- No API keys needed

**Usage:**
```bash
python simple_api_test.py
```

## Running the Tests

### Prerequisites
1. **Start the server:**
   ```bash
   python main.py
   ```

2. **Set up environment variables** (for full test only):
   ```bash
   # In your .env file
   OPENAI_API_KEY=your_openai_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Running Tests

**Quick API validation:**
```bash
python simple_api_test.py
```

**Full end-to-end test:**
```bash
python test_interview_endpoints.py
```

## Test Flow

Both scripts follow the same test flow:

1. **Health Check** - Verify server is running
2. **Start Interview** - Create new session and get first question
3. **Answer Questions** - Simulate candidate responses until interview complete
4. **Get Evaluation** - Retrieve final score and feedback

## Expected Output

### Successful Test Run:
```
🧪 Interview API Test Suite
==================================================
✅ Server is running: {'message': 'Interview Chatbot API is running', 'version': '1.0.0'}

🔍 Testing health endpoint...
✅ Health check passed: {'status': 'healthy', 'service': 'interview-chatbot'}

🚀 Testing start-interview endpoint for role: Data Scientist
✅ Interview started successfully!
   Session ID: 12345678-1234-1234-1234-123456789abc
   Question #1: Could you start by telling me about yourself...
   Audio length: 12345 characters (base64)

🎤 Testing answer-question endpoint...
✅ Answer processed successfully!
   Next Question #2: Can you walk me through a challenging...

[... continues until interview complete ...]

📊 Testing get-evaluation endpoint...
✅ Evaluation retrieved successfully!
   Overall Score: 8/10
   Feedback: The candidate demonstrated strong technical knowledge...

🎉 COMPLETE INTERVIEW TEST PASSED!
```

## Audio Samples

The full test script saves audio samples in `./audio_samples/` directory:
- `question_1.wav` - First interview question
- `question_2.wav` - Second interview question  
- etc.

You can play these files to verify the TTS quality.

## Troubleshooting

### Common Issues:

1. **Server not accessible**
   - Make sure `python main.py` is running
   - Check if port 8000 is available

2. **API key errors** (full test only)
   - Verify `.env` file contains valid API keys
   - Check API key permissions and quotas

3. **Audio processing errors** (full test only)
   - Ensure OpenAI API key has access to TTS/STT
   - Check internet connectivity

4. **Import errors**
   - Install requirements: `pip install -r requirements.txt`

### Debug Mode:
Add print statements or use the simple test first to isolate issues:

```bash
# Test API logic only
python simple_api_test.py

# If that works, then try full test
python test_interview_endpoints.py
```

## Test Configuration

You can modify the test scripts to:

- Change the job role: Edit `job_role` parameter
- Add custom test answers: Modify `test_answers` list  
- Adjust question limits: Change `max_questions` variable
- Test different TTS voices: Modify voice parameter in audio generation

## Integration with CI/CD

The simple test script is suitable for automated testing:

```bash
# In your CI pipeline
python simple_api_test.py
if [ $? -eq 0 ]; then
    echo "API tests passed"
else
    echo "API tests failed"
    exit 1
fi
```