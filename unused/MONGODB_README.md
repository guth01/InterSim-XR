# Interview API with MongoDB Atlas Integration

## 🎯 Overview

This is an enhanced version of the Interview Chatbot API that includes:
- **5-digit access codes** for easy session management
- **MongoDB Atlas integration** for persistent data storage
- **Voice metrics analysis** for speech pattern evaluation
- **Confidence scoring** using OpenAI analysis
- **Report generation** combining interview and voice data

## 🏗️ System Architecture

### Database Structure (MongoDB Atlas)
```
Collection: interviews
Document Structure:
{
  "_id": "12345",                    // 5-digit access code (primary key)
  "access_code": "12345",
  "job_role": "Data Scientist",
  "job_description": "Optional description",
  "resume_text": "Optional resume content",
  "qa_pairs": [
    {
      "question": "Tell me about yourself",
      "answer": "I'm a data scientist...",
      "question_number": 1,
      "voice_metrics": {
        "avg_confidence": 0.85,
        "speech_rate": 2.3,
        "avg_pause": 0.4,
        "pitch_mean": 150.2,
        "pitch_std": 25.1,
        "energy_mean": 0.6,
        "energy_std": 0.15
      },
      "timestamp": "2025-09-21T10:30:00"
    }
  ],
  "interview_complete": false,
  "created_at": "2025-09-21T10:00:00",
  "current_question_index": 0,
  "initial_questions": ["Question 1", "Question 2", ...]
}
```

## 🛠️ API Endpoints

### 1. **POST /interview/setup-interview**
Creates a new interview session and returns a 5-digit access code.

**Request:**
```json
{
  "job_role": "Data Scientist",
  "job_description": "Optional job description text",
  "resume_text": "Optional resume content"
}
```

**Response:**
```json
{
  "access_code": "12345",
  "message": "Interview session created successfully. Use access code 12345 to start your interview."
}
```

### 2. **POST /interview/start-interview**
Starts the interview using the 5-digit access code.

**Request:**
```json
{
  "access_code": "12345"
}
```

**Response:**
```json
{
  "access_code": "12345",
  "question_text": "Could you start by telling me about yourself?",
  "question_audio": "base64-encoded-audio-data",
  "question_number": 1
}
```

### 3. **POST /interview/answer-question**
Processes candidate's audio answer, calculates voice metrics, and returns next question.

**Request:**
```json
{
  "access_code": "12345",
  "audio_data": "base64-encoded-wav-audio"
}
```

**Response:**
```json
{
  "question_text": "Next question text or null if complete",
  "question_audio": "base64-encoded-audio or null",
  "question_number": 2,
  "interview_completed": false
}
```

**Voice Metrics Calculated:**
- Average Whisper confidence score
- Speech rate (words/second)
- Average pause duration between words
- Pitch analysis (mean and standard deviation)
- Energy analysis (mean and standard deviation)

### 4. **POST /interview/generate-report**
Generates comprehensive interview report with confidence scoring (only available after interview completion).

**Request:**
```json
{
  "access_code": "12345"
}
```

**Response:**
```json
{
  "access_code": "12345",
  "confidence_score": 0.82,
  "detailed_analysis": "The candidate demonstrated strong technical knowledge...",
  "voice_metrics_summary": {
    "avg_confidence": 0.85,
    "avg_speech_rate": 2.3,
    "avg_pause": 0.4,
    "pitch_consistency": 0.92,
    "energy_consistency": 0.88
  },
  "qa_summary": {
    "total_questions": 6,
    "avg_answer_length": 45.2,
    "total_interview_duration": "Estimated based on questions answered"
  }
}
```

### 5. **GET /interview/session/{access_code}**
Get current interview session status.

**Response:**
```json
{
  "access_code": "12345",
  "job_role": "Data Scientist",
  "questions_answered": 3,
  "interview_complete": false,
  "created_at": "2025-09-21T10:00:00"
}
```

### 6. **GET /interview/health**
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "interview-chatbot-mongo"
}
```

## 🎵 Voice Metrics Analysis

The system analyzes the following voice characteristics from candidate audio:

1. **Confidence Score** - Based on Whisper transcription confidence
2. **Speech Rate** - Words per second calculation
3. **Pause Analysis** - Average silence duration between words
4. **Pitch Analysis** - Fundamental frequency statistics
5. **Energy Analysis** - Audio signal energy measurements

These metrics are used to assess:
- Speaking confidence and fluency
- Communication clarity
- Nervousness indicators
- Overall interview performance

## 📊 Confidence Scoring System

The report generation uses OpenAI GPT-4 to analyze:
- **Content Quality** - Relevance and depth of answers
- **Communication Skills** - Clarity and structure
- **Voice Metrics** - Speech patterns and confidence indicators
- **Technical Knowledge** - Domain-specific expertise demonstration

**Confidence Score Scale:**
- `0.0-0.3`: Low confidence/poor performance
- `0.4-0.6`: Moderate confidence/average performance
- `0.7-0.8`: Good confidence/strong performance
- `0.9-1.0`: Excellent confidence/outstanding performance

## 🚀 Setup and Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. MongoDB Atlas Setup
1. Create a MongoDB Atlas account
2. Create a new cluster
3. Get your connection string
4. Create a database called `interview_db`

### 3. Environment Configuration
Copy `.env.example` to `.env` and configure:
```env
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/interview_db?retryWrites=true&w=majority
```

### 4. Run the Server
```bash
python main_mongo.py
```

## 🧪 Testing

### Quick Test
```bash
python test_mongo_api.py
```

This test will:
1. Create a new interview session (get 5-digit code)
2. Start the interview
3. Answer all questions with mock audio
4. Generate a comprehensive report
5. Verify all voice metrics are calculated

### Manual Testing with curl

**Setup Interview:**
```bash
curl -X POST "http://localhost:8000/interview/setup-interview" \
  -H "Content-Type: application/json" \
  -d '{
    "job_role": "Data Scientist",
    "job_description": "Senior role requiring ML expertise"
  }'
```

**Start Interview:**
```bash
curl -X POST "http://localhost:8000/interview/start-interview" \
  -H "Content-Type: application/json" \
  -d '{"access_code": "12345"}'
```

## 🔄 Workflow Example

1. **Candidate Setup**
   - POST `/setup-interview` with job details
   - Receive 5-digit access code (e.g., "67890")

2. **Interview Start**
   - POST `/start-interview` with access code
   - Receive first question + audio

3. **Answer Loop**
   - POST `/answer-question` with access code + audio
   - System calculates voice metrics and stores in MongoDB
   - Receive next question or completion signal

4. **Report Generation**
   - POST `/generate-report` with access code
   - Receive comprehensive analysis with confidence score

## 🔐 Security Considerations

- Access codes are 5-digit random numbers (100,000 combinations)
- Codes are unique and checked against existing database entries
- Session data is stored securely in MongoDB Atlas
- API keys should be kept secure and not exposed

## 📈 Monitoring and Analytics

The system stores comprehensive data for analytics:
- Interview completion rates
- Average voice metrics by job role
- Question difficulty analysis
- Performance trends over time

## 🚀 Production Deployment

For production use:
1. Configure MongoDB Atlas with appropriate security
2. Set up proper CORS policies
3. Use environment-specific configuration
4. Implement rate limiting
5. Add authentication if needed
6. Monitor database performance and costs

## 🔧 Customization

### Adding New Voice Metrics
Modify `calculate_voice_metrics()` function in `interview_router_mongo.py`

### Customizing Questions
Questions are generated by OpenAI GPT-4. Modify the prompt in `generate_initial_questions()`

### Adjusting Confidence Scoring
Update the analysis prompt in `generate_confidence_report()`

## 📝 API Response Examples

See `test_mongo_api.py` for complete request/response examples and testing patterns.