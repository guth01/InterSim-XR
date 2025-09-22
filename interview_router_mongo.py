from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import base64
import io
import json
import random
import string
import gc
from datetime import datetime
from openai import OpenAI
import google.generativeai as genai
import os
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
import whisper
import tempfile
import traceback
import os

# Initialize clients
openai_client = OpenAI()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Global variable for lazy loading Whisper model
whisper_model = None
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")  # tiny, base, small, medium, large
DISABLE_WHISPER = os.getenv("DISABLE_WHISPER", "false").lower() == "true"

def load_whisper_model():
    """Lazy load Whisper model only when needed"""
    global whisper_model
    
    if DISABLE_WHISPER:
        print("[Whisper] Whisper disabled via environment variable")
        return None
        
    if whisper_model is None:
        try:
            print(f"[Whisper] Loading Whisper model ({WHISPER_MODEL_SIZE})...")
            whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
            print("[Whisper] Model loaded successfully!")
        except Exception as e:
            print(f"[Whisper] Failed to load model: {e}")
            whisper_model = False  # Mark as failed to avoid repeated attempts
    
    return whisper_model if whisper_model is not False else None

router = APIRouter(prefix="/interview", tags=["interview"])

# ===== MongoDB Connection =====
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = "interview_db"
COLLECTION_NAME = "interviews"

# Initialize MongoDB clients with error handling
mongo_client = None
db = None
interviews_collection = None
async_mongo_client = None
async_db = None
async_interviews_collection = None

# ===== Section Constants =====

SECTION_NAMES = {
    1: "General Introduction",
    2: "Technical & Role-based",
    3: "Resume & Experience",
    4: "Behavioral Assessment"
}

SECTION_QUESTION_COUNTS = {
    1: 2,
    2: 3,
    3: 3,
    4: 3
}

# Predefined behavioral questions bank
BEHAVIORAL_QUESTIONS = [
    "Tell me about a time when you had to work under a tight deadline. How did you manage it?",
    "Describe a situation where you had to work with a difficult team member. How did you handle it?",
    "Can you share an example of a time when you had to learn something new quickly?",
    "Tell me about a project that didn't go as planned. What did you do?",
    "Describe a time when you had to give constructive feedback to a colleague.",
    "Can you tell me about a time when you took initiative on a project?",
    "Describe a situation where you had to adapt to a significant change at work.",
    "Tell me about a time when you made a mistake. How did you handle it?",
    "Can you share an example of when you had to persuade someone to see things your way?",
    "Describe a time when you went above and beyond your job responsibilities.",
    "Tell me about a challenging problem you solved creatively.",
    "Can you describe a time when you had to work with limited resources?",
    "Tell me about a time when you had to manage multiple priorities.",
    "Describe a situation where you had to deal with ambiguous requirements.",
    "Can you share an example of when you mentored or helped train someone?"
]

def init_mongodb():
    """Initialize MongoDB connection with error handling"""
    global mongo_client, db, interviews_collection, async_mongo_client, async_db, async_interviews_collection
    
    try:
        print(f"[MongoDB] Attempting to connect to: {MONGODB_URL[:50]}...")
        
        # Test connection first
        test_client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        test_client.admin.command('ping')
        print("[MongoDB] Connection test successful!")
        
        # Initialize clients
        mongo_client = MongoClient(MONGODB_URL)
        db = mongo_client[DATABASE_NAME]
        interviews_collection = db[COLLECTION_NAME]
        
        # Async client
        async_mongo_client = AsyncIOMotorClient(MONGODB_URL)
        async_db = async_mongo_client[DATABASE_NAME]
        async_interviews_collection = async_db[COLLECTION_NAME]
        
        print(f"[MongoDB] Successfully connected to database: {DATABASE_NAME}")
        return True
        
    except Exception as e:
        print(f"[MongoDB] Connection failed: {e}")
        print("[MongoDB] Falling back to in-memory storage...")
        return False

# Try to initialize MongoDB
mongodb_available = init_mongodb()

# Fallback in-memory storage
memory_storage = {} if not mongodb_available else None

# ===== DTOs =====

class SetupInterviewIn(BaseModel):
    job_role: str = "Data Scientist"
    job_description: Optional[str] = None
    resume_text: Optional[str] = None

class SetupInterviewOut(BaseModel):
    access_code: str  # 5-digit code
    message: str

class StartInterviewIn(BaseModel):
    access_code: str  # 5-digit code

class StartInterviewOut(BaseModel):
    access_code: str
    question_text: str
    question_audio: str  # base64 encoded audio
    question_number: int
    section: int
    section_name: str
    total_questions: int

class AnswerQuestionIn(BaseModel):
    access_code: str
    audio_data: str  # base64 encoded WAV

class AnswerQuestionOut(BaseModel):
    question_text: Optional[str] = None
    question_audio: Optional[str] = None
    question_number: Optional[int] = None
    section: Optional[int] = None
    section_name: Optional[str] = None
    total_questions: Optional[int] = None
    interview_completed: bool = False

class GenerateReportIn(BaseModel):
    access_code: str

class GenerateReportOut(BaseModel):
    access_code: str
    confidence_score: float  # 0.0 to 1.0
    detailed_analysis: str
    voice_metrics_summary: Dict[str, Any]
    qa_summary: Dict[str, Any]

class GetEvaluationIn(BaseModel):
    access_code: str
    section: int  # 2, 3, or 4 (section 1 not evaluated)

class GetEvaluationOut(BaseModel):
    section: int
    section_name: str
    overall_score: int  # 1-10
    feedback: str
    transcript: List[Dict[str, str]]  # List of Q&A pairs for that section

class VoiceMetrics(BaseModel):
    avg_confidence: float
    speech_rate: float
    avg_pause: float
    pitch_mean: float
    pitch_std: float
    energy_mean: float
    energy_std: float

class QAPair(BaseModel):
    question: str
    answer: str
    question_number: int
    section: int
    voice_metrics: VoiceMetrics
    timestamp: datetime

# ===== MongoDB Document Models =====

class InterviewDocument:
    """MongoDB document structure for interviews"""
    def __init__(self, access_code: str, job_role: str, job_description: str = None, resume_text: str = None):
        self.access_code = access_code
        self.job_role = job_role
        self.job_description = job_description
        self.resume_text = resume_text
        self.qa_pairs = []
        self.interview_complete = False
        self.created_at = datetime.now()
        self.current_section = 1
        self.current_question_in_section = 0
        self.section_questions = {"1": [], "2": [], "3": [], "4": []}  # Store questions by section (string keys for MongoDB)
    
    def to_dict(self):
        return {
            "_id": self.access_code,
            "access_code": self.access_code,
            "job_role": self.job_role,
            "job_description": self.job_description,
            "resume_text": self.resume_text,
            "qa_pairs": self.qa_pairs,
            "interview_complete": self.interview_complete,
            "created_at": self.created_at,
            "current_section": self.current_section,
            "current_question_in_section": self.current_question_in_section,
            "section_questions": self.section_questions
        }

# ===== Helper Functions =====

def generate_access_code() -> str:
    """Generate a unique 5-digit access code"""
    while True:
        code = ''.join(random.choices(string.digits, k=5))
        
        # Check if code already exists
        if mongodb_available:
            if not interviews_collection.find_one({"_id": code}):
                return code
        else:
            if code not in memory_storage:
                return code

def store_interview_document(interview_doc_dict: dict) -> bool:
    """Store interview document in database or memory"""
    try:
        if mongodb_available:
            interviews_collection.insert_one(interview_doc_dict)
        else:
            memory_storage[interview_doc_dict["_id"]] = interview_doc_dict
        return True
    except Exception as e:
        print(f"[Storage] Error storing document: {e}")
        return False

def get_interview_document(access_code: str) -> dict:
    """Get interview document from database or memory"""
    try:
        if mongodb_available:
            return interviews_collection.find_one({"_id": access_code})
        else:
            return memory_storage.get(access_code)
    except Exception as e:
        print(f"[Storage] Error retrieving document: {e}")
        return None

def update_interview_document(access_code: str, update_data: dict) -> bool:
    """Update interview document in database or memory with enhanced error handling"""
    try:
        if mongodb_available:
            print(f"[Storage] Updating MongoDB document {access_code} with: {list(update_data.keys())}")
            
            # Use upsert=False to ensure document exists
            result = interviews_collection.update_one(
                {"_id": access_code},
                update_data,
                upsert=False
            )
            
            if result.matched_count == 0:
                print(f"[Storage] ERROR: No document found with access_code: {access_code}")
                return False
            elif result.modified_count == 0:
                print(f"[Storage] WARNING: Document found but no changes made")
                return True
            else:
                print(f"[Storage] Successfully updated document. Modified: {result.modified_count}")
                return True
                
        else:
            # Memory storage handling (unchanged)
            if access_code in memory_storage:
                if "$set" in update_data:
                    memory_storage[access_code].update(update_data["$set"])
                if "$push" in update_data:
                    for key, value in update_data["$push"].items():
                        if key not in memory_storage[access_code]:
                            memory_storage[access_code][key] = []
                        memory_storage[access_code][key].append(value)
                return True
            else:
                print(f"[Storage] ERROR: Document not found in memory: {access_code}")
                return False
                
    except Exception as e:
        print(f"[Storage] Error updating document: {e}")
        import traceback
        traceback.print_exc()
        return False


def calculate_voice_metrics(transcription_result: str, audio_data: bytes) -> Dict[str, float]:
    """Calculate voice metrics from audio data using Whisper for confidence"""
    try:
        # Save audio bytes to temporary file for Whisper processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name
        
        # Initialize default metrics
        metrics = {
            "avg_confidence": 0.8,  # fallback
            "speech_rate": 2.0,
            "avg_pause": 0.2,
            "pitch_mean": 150.0,
            "pitch_std": 20.0,
            "energy_mean": 0.5,
            "energy_std": 0.1
        }
        
        # Use Whisper for confidence calculation if available
        whisper_model = load_whisper_model()
        if whisper_model is not None:
            try:
                print("[Voice Metrics] Using Whisper for confidence calculation...")
                
                # Transcribe with word-level timestamps and confidence
                result = whisper_model.transcribe(
                    temp_audio_path, 
                    word_timestamps=True,
                    verbose=False
                )
                
                # Extract confidence from Whisper segments
                word_confidences = []
                word_times = []
                
                for segment in result.get('segments', []):
                    # Segment-level confidence from avg_logprob
                    if 'avg_logprob' in segment:
                        segment_confidence = np.exp(segment['avg_logprob'])
                        word_confidences.append(segment_confidence)
                    
                    # Extract word-level timing for pause calculation
                    for word in segment.get('words', []):
                        if 'start' in word and 'end' in word:
                            word_times.append((float(word['start']), float(word['end'])))
                        
                        # Word-level confidence if available
                        if 'probability' in word:
                            word_confidences.append(float(word['probability']))
                        elif 'avg_logprob' in word:
                            word_confidence = np.exp(word['avg_logprob'])
                            word_confidences.append(word_confidence)
                
                # Calculate average confidence
                if word_confidences:
                    metrics["avg_confidence"] = float(np.mean(word_confidences))
                    print(f"[Voice Metrics] Whisper confidence: {metrics['avg_confidence']:.3f}")
                
                # Calculate speech rate from Whisper timing
                if result.get('segments'):
                    total_words = sum(len(seg.get('words', [])) for seg in result['segments'])
                    if total_words > 0:
                        first_start = float(result['segments'][0].get('start', 0))
                        last_end = float(result['segments'][-1].get('end', 0))
                        total_duration = last_end - first_start
                        
                        if total_duration > 0:
                            metrics["speech_rate"] = total_words / total_duration
                
                # Calculate pause duration between words
                if len(word_times) > 1:
                    pauses = []
                    for i in range(len(word_times) - 1):
                        pause_duration = word_times[i + 1][0] - word_times[i][1]
                        if pause_duration > 0:  # Only positive pauses
                            pauses.append(pause_duration)
                    
                    if pauses:
                        metrics["avg_pause"] = float(np.mean(pauses))
                
                print(f"[Voice Metrics] Whisper speech rate: {metrics['speech_rate']:.2f} words/sec")
                print(f"[Voice Metrics] Whisper avg pause: {metrics['avg_pause']:.3f} sec")
                
            except Exception as e:
                print(f"[Voice Metrics] Whisper processing failed: {e}")
                # Continue with librosa processing for other metrics
        
        # Use librosa for additional audio features
        try:
            audio_io = io.BytesIO(audio_data)
            audio_array, sample_rate = librosa.load(audio_io, sr=None)
            
            # Calculate energy features
            frame_length = 2048
            hop_length = 512
            energy = np.array([
                sum(abs(audio_array[i:i+frame_length]**2)) 
                for i in range(0, len(audio_array), hop_length)
            ])
            
            if len(energy) > 0:
                metrics["energy_mean"] = float(np.mean(energy))
                metrics["energy_std"] = float(np.std(energy))
            
            # Calculate pitch features using librosa
            try:
                fmin = librosa.note_to_hz('C2')  # ~65 Hz
                fmax = librosa.note_to_hz('C7')  # ~2093 Hz
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    audio_array, 
                    fmin=fmin, 
                    fmax=fmax, 
                    sr=sample_rate
                )
                
                # Filter out NaN values and extract valid pitch values
                valid_f0 = f0[~np.isnan(f0)]
                if len(valid_f0) > 0:
                    metrics["pitch_mean"] = float(np.mean(valid_f0))
                    metrics["pitch_std"] = float(np.std(valid_f0))
                
            except Exception as e:
                print(f"[Voice Metrics] Pitch calculation failed: {e}")
                # Keep default pitch values
            
        except Exception as e:
            print(f"[Voice Metrics] Librosa processing failed: {e}")
            # Keep default values
        
        # Clean up temporary file
        try:
            os.unlink(temp_audio_path)
        except:
            pass
        
        # Force garbage collection to free memory
        gc.collect()
        
        # Ensure all values are within reasonable ranges
        metrics["avg_confidence"] = max(0.0, min(1.0, metrics["avg_confidence"]))
        metrics["speech_rate"] = max(0.0, min(10.0, metrics["speech_rate"]))
        metrics["avg_pause"] = max(0.0, min(5.0, metrics["avg_pause"]))
        metrics["pitch_mean"] = max(50.0, min(500.0, metrics["pitch_mean"]))
        
        print(f"[Voice Metrics] Final metrics: confidence={metrics['avg_confidence']:.3f}, "
                f"speech_rate={metrics['speech_rate']:.2f}, pause={metrics['avg_pause']:.3f}")
        
        return metrics
        
    except Exception as e:
        print(f"[Voice Metrics] Error calculating metrics: {e}")
        traceback.print_exc()
        
        # Force garbage collection even on error
        gc.collect()
        
        # Return safe default values if everything fails
        return {
            "avg_confidence": 0.7,
            "speech_rate": 2.0,
            "avg_pause": 0.3,
            "pitch_mean": 150.0,
            "pitch_std": 25.0,
            "energy_mean": 0.4,
            "energy_std": 0.15
        }

def transcribe_audio(base64_audio_data: str) -> str:
    """Transcribes base64 audio using OpenAI Whisper"""
    try:
        audio_bytes = base64.b64decode(base64_audio_data)
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav"
        
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcription.text
    except Exception as e:
        print(f"[Whisper] Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {e}")

def text_to_speech(text: str) -> str:
    """Converts text to speech and returns base64 encoded audio"""
    try:
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text,
            response_format="wav"
        )
        
        audio_bytes = response.content
        return base64.b64encode(audio_bytes).decode('utf-8')
    except Exception as e:
        print(f"[TTS] Text-to-speech error: {e}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {e}")

# def generate_initial_questions(job_role: str, job_description: Optional[str] = None) -> List[str]:
#     """Returns hardcoded interview questions (no LLM generation for reliable testing)"""
#     print(f"[Questions] Using hardcoded questions for {job_role}")
    
#     # Always return the same 6 questions for consistent testing
#     return [
#         f"Could you start by telling me about yourself and what interests you about the {job_role} role?",
#         "Can you walk me through a challenging technical project you've worked on recently?",
#         "How do you approach problem-solving when faced with a technical issue you haven't encountered before?",
#         "What tools and technologies are you most comfortable working with, and why?",
#         "Can you describe a time when you had to learn a new technology or skill quickly?",
#         "What do you see as the biggest challenges in the field right now, and how do you stay current?"
#     ]

def generate_all_section_questions(job_role: str, job_description: Optional[str] = None, resume_text: Optional[str] = None) -> Dict[str, List[str]]:
    """Generate questions for all sections at session creation"""
    return {
        "1": generate_general_questions(),
        "2": generate_role_questions(job_role, job_description),
        "3": generate_resume_questions(job_role, resume_text),
        "4": select_behavioral_questions()
    }

def generate_general_questions() -> List[str]:
    """Generate 2 general introduction questions"""
    return [
        "Could you please introduce yourself and tell me a bit about your background?",
        "What motivated you to apply for this position and what interests you about our company?"
    ]

def generate_role_questions(job_role: str, job_description: Optional[str] = None) -> List[str]:
    """Generate 3 role-specific technical questions"""
    description_context = f"\nJob Description: {job_description}" if job_description else ""
    
    system_prompt = f"""You are an expert interviewer conducting a technical interview for a {job_role} role.{description_context}

Generate exactly 3 technical interview questions that:
1. Cover core technical skills relevant to the role
2. Include problem-solving scenarios
3. Assess depth of technical knowledge

Return ONLY the questions, numbered 1-3, one per line."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use gpt-3.5-turbo instead of gpt-4
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate 3 technical questions for {job_role}"}
            ],
            temperature=0.7
        )
        
        questions_text = response.choices[0].message.content
        questions = [q.strip() for q in questions_text.split('\n') if q.strip() and any(c.isdigit() for c in q[:3])]
        
        # Clean up numbering
        cleaned_questions = []
        for q in questions:
            clean_q = q.split('.', 1)[-1].strip() if '.' in q[:5] else q
            cleaned_questions.append(clean_q)
        
        return cleaned_questions[:3]
        
    except Exception as e:
        print(f"[OpenAI] Role question generation error: {e}")
        # Fallback questions
        return [
            f"What are the most important technical skills for a {job_role} and how do you rate yourself in each?",
            "Can you walk me through your approach to solving a complex technical problem?",
            "What tools and technologies do you prefer working with and why?"
        ]

def generate_resume_questions(job_role: str, resume_text: Optional[str] = None) -> List[str]:
    """Generate 3 resume-specific questions"""
    if not resume_text:
        # Fallback questions if no resume provided
        return [
            "Can you tell me about your most significant professional achievement?",
            "Describe a challenging project from your previous experience.",
            "How has your career progression prepared you for this role?"
        ]
    
    system_prompt = f"""You are an interviewer reviewing a candidate's resume for a {job_role} position.

Resume Content:
{resume_text[:2000]}...

Generate exactly 3 questions that:
1. Focus on specific experiences mentioned in the resume
2. Ask for elaboration on key projects or achievements
3. Connect past experience to the current role

Return ONLY the questions, numbered 1-3, one per line."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use gpt-3.5-turbo instead of gpt-4
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate 3 resume-based questions"}
            ],
            temperature=0.7
        )
        
        questions_text = response.choices[0].message.content
        questions = [q.strip() for q in questions_text.split('\n') if q.strip() and any(c.isdigit() for c in q[:3])]
        
        # Clean up numbering
        cleaned_questions = []
        for q in questions:
            clean_q = q.split('.', 1)[-1].strip() if '.' in q[:5] else q
            cleaned_questions.append(clean_q)
        
        return cleaned_questions[:3]
        
    except Exception as e:
        print(f"[OpenAI] Resume question generation error: {e}")
        # Fallback questions
        return [
            "Can you elaborate on one of the key projects mentioned in your resume?",
            "How do your previous roles connect to what you'd be doing in this position?",
            "What was your biggest learning from your most recent position?"
        ]

def select_behavioral_questions() -> List[str]:
    """Select 3 random behavioral questions from the bank"""
    return random.sample(BEHAVIORAL_QUESTIONS, 3)

def get_current_question_info(interview_doc: dict) -> tuple[str, int, int, str]:
    """Returns (question_text, question_number, section, section_name)"""
    current_section = interview_doc["current_section"]
    current_q_index = interview_doc["current_question_in_section"]
    
    # Always use string keys for MongoDB compatibility
    section_questions = interview_doc["section_questions"][str(current_section)]
    
    if current_q_index >= len(section_questions):
        raise HTTPException(status_code=500, detail="No more questions in current section")
    
    question_text = section_questions[current_q_index]
    question_number = current_q_index + 1
    section_name = SECTION_NAMES[current_section]
    
    return question_text, question_number, current_section, section_name

def advance_to_next_question(access_code: str, interview_doc: dict) -> bool:
    """Advances to next question. Returns True if there are more questions, False if interview complete"""
    current_section = interview_doc["current_section"]
    current_q_index = interview_doc["current_question_in_section"]
    
    # Move to next question in current section
    new_q_index = current_q_index + 1
    
    # Always use string keys for MongoDB compatibility
    section_questions = interview_doc["section_questions"][str(current_section)]
    
    # Check if we've finished current section
    if new_q_index >= len(section_questions):
        # Move to next section
        if current_section < 4:
            new_section = current_section + 1
            new_q_index = 0
            print(f"[Interview] Advanced to section {new_section}")
            
            update_interview_document(
                access_code,
                {"$set": {
                    "current_section": new_section,
                    "current_question_in_section": new_q_index
                }}
            )
            return True
        else:
            # Interview complete
            update_interview_document(
                access_code,
                {"$set": {
                    "interview_complete": True,
                    "current_question_in_section": new_q_index
                }}
            )
            print(f"[Interview] Interview completed")
            return False
    else:
        # Stay in current section, advance question
        update_interview_document(
            access_code,
            {"$set": {"current_question_in_section": new_q_index}}
        )
        return True

def get_section_qa_pairs(interview_doc: dict, section: int) -> List[Dict]:
    """Get Q&A pairs for a specific section"""
    return [qa for qa in interview_doc.get("qa_pairs", []) if qa.get("section") == section]

class GeminiClient:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def evaluate_section(self, job_role: str, section: int, qa_pairs: List[Dict]) -> Dict[str, Any]:
        """Evaluate a specific section of the interview"""
        section_name = SECTION_NAMES[section]
        
        # Prepare transcript for evaluation
        transcript = ""
        for i, qa in enumerate(qa_pairs, 1):
            transcript += f"Q{i}: {qa['question']}\nA{i}: {qa['answer']}\n\n"
        
        # Section-specific evaluation prompts
        if section == 2:  # Technical/Role-based
            focus_areas = "technical knowledge, problem-solving ability, and role-specific skills"
        elif section == 3:  # Resume-based
            focus_areas = "relevant experience, project depth, and career progression alignment"
        elif section == 4:  # Behavioral
            focus_areas = "behavioral competencies, situational judgment, and soft skills"
        else:
            focus_areas = "overall performance"
        
        evaluation_prompt = f"""
        You are an expert interviewer evaluating the {section_name} section of a candidate interview for a {job_role} position.
        
        {section_name} Section Transcript:
        {transcript}
        
        Evaluate this section focusing on: {focus_areas}
        
        Please provide:
        1. A section-specific score from 1-10 (where 10 is excellent)
        2. Detailed feedback specific to this section's focus areas
        
        Consider:
        - Quality and depth of responses
        - Relevance to the section's purpose
        - Communication clarity
        - Specific strengths and areas for improvement
        
        Format your response as:
        SCORE: [number]
        FEEDBACK: [detailed feedback]
        """
        
        try:
            response = self.model.generate_content(evaluation_prompt)
            result_text = response.text
            
            # Parse score and feedback
            lines = result_text.strip().split('\n')
            score = 7  # default
            feedback = result_text
            
            for line in lines:
                if line.startswith('SCORE:'):
                    try:
                        score = int(line.split(':')[1].strip())
                    except:
                        score = 7
                elif line.startswith('FEEDBACK:'):
                    feedback = line.split(':', 1)[1].strip()
            
            return {
                "overall_score": max(1, min(10, score)),
                "feedback": feedback
            }
        except Exception as e:
            print(f"[Gemini] Section {section} evaluation failed: {e}")
            return {
                "overall_score": 7,
                "feedback": f"{section_name} section completed successfully. {len(qa_pairs)} questions answered. Due to evaluation service unavailability, a detailed analysis cannot be provided at this time."
            }

# Initialize Gemini client
gemini_client = GeminiClient()

def ensure_serializable_qa_pair(qa_pair: dict) -> dict:
    """Ensure QA pair is serializable for MongoDB storage"""
    serializable_qa = qa_pair.copy()
    
    # Ensure all voice metrics are float (not numpy types)
    if "voice_metrics" in serializable_qa:
        vm = serializable_qa["voice_metrics"]
        for key, value in vm.items():
            if hasattr(value, 'item'):  # numpy scalar
                vm[key] = float(value.item())
            elif isinstance(value, (int, float)):
                vm[key] = float(value)
    
    return serializable_qa

def generate_confidence_report(qa_pairs: List[Dict], voice_metrics: List[Dict]) -> Dict[str, Any]:
    """Generate confidence report using OpenAI"""
    # Prepare data for analysis
    qa_text = ""
    for i, qa in enumerate(qa_pairs, 1):
        qa_text += f"Q{i}: {qa['question']}\nA{i}: {qa['answer']}\n\n"
    
    # Calculate average voice metrics
    if voice_metrics:
        avg_metrics = {
            "avg_confidence": np.mean([m["avg_confidence"] for m in voice_metrics]),
            "avg_speech_rate": np.mean([m["speech_rate"] for m in voice_metrics]),
            "avg_pause": np.mean([m["avg_pause"] for m in voice_metrics]),
            "pitch_consistency": 1.0 - (np.std([m["pitch_mean"] for m in voice_metrics]) / 100),
            "energy_consistency": 1.0 - (np.std([m["energy_mean"] for m in voice_metrics]) / 10)
        }
    else:
        avg_metrics = {"avg_confidence": 0.5, "avg_speech_rate": 0.5, "avg_pause": 0.5, "pitch_consistency": 0.5, "energy_consistency": 0.5}
    
    system_prompt = """You are an expert interview analyst. Analyze the interview transcript and voice metrics to provide a comprehensive confidence score and analysis.

Consider:
1. Content quality and relevance of answers
2. Communication clarity and structure
3. Voice confidence indicators (speech rate, pauses, pitch consistency)
4. Technical knowledge demonstration
5. Overall interview performance

Provide a confidence score from 0.0 to 1.0 where:
- 0.0-0.3: Low confidence/poor performance
- 0.4-0.6: Moderate confidence/average performance  
- 0.7-0.8: Good confidence/strong performance
- 0.9-1.0: Excellent confidence/outstanding performance

Format your response as:
CONFIDENCE_SCORE: [0.0-1.0]
ANALYSIS: [detailed analysis]"""

    try:
        user_prompt = f"""Interview Transcript:
{qa_text}

Voice Metrics Summary:
- Average Confidence: {avg_metrics['avg_confidence']:.3f}
- Average Speech Rate: {avg_metrics['avg_speech_rate']:.2f} words/sec
- Average Pause Duration: {avg_metrics['avg_pause']:.3f} sec
- Pitch Consistency: {avg_metrics['pitch_consistency']:.3f}
- Energy Consistency: {avg_metrics['energy_consistency']:.3f}

Analyze this interview performance and provide a confidence score."""

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use gpt-3.5-turbo instead of gpt-4
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        
        result_text = response.choices[0].message.content
        
        # Parse confidence score and analysis
        confidence_score = 0.7  # default
        analysis = result_text
        
        lines = result_text.strip().split('\n')
        for line in lines:
            if line.startswith('CONFIDENCE_SCORE:'):
                try:
                    confidence_score = float(line.split(':')[1].strip())
                    confidence_score = max(0.0, min(1.0, confidence_score))
                except:
                    confidence_score = 0.7
            elif line.startswith('ANALYSIS:'):
                analysis = line.split(':', 1)[1].strip()
        
        return {
            "confidence_score": confidence_score,
            "detailed_analysis": analysis,
            "voice_metrics_summary": avg_metrics
        }
        
    except Exception as e:
        print(f"[Report Generation] Error: {e}")
        return {
            "confidence_score": 0.7,
            "detailed_analysis": "Report generation temporarily unavailable. Based on interview completion, candidate showed reasonable performance.",
            "voice_metrics_summary": avg_metrics
        }

# ===== API Endpoints =====

@router.post("/setup-interview", response_model=SetupInterviewOut)
def setup_interview(inp: SetupInterviewIn):
    """Creates a new interview session and returns a 5-digit access code"""
    print(f"[Setup] Creating 4-section interview session for role: {inp.job_role}")
    print(f"[Setup] Storage mode: {'MongoDB' if mongodb_available else 'In-Memory'}")
    
    # Generate unique access code
    access_code = generate_access_code()
    
    # Create interview document
    interview_doc = InterviewDocument(
        access_code=access_code,
        job_role=inp.job_role,
        job_description=inp.job_description,
        resume_text=inp.resume_text
    )
    
    # Generate all section questions at setup
    section_questions = generate_all_section_questions(
        inp.job_role, 
        inp.job_description, 
        inp.resume_text
    )
    interview_doc.section_questions = section_questions
    
    print(f"[Setup] Generated questions for all 4 sections:")
    for section_num, questions in section_questions.items():
        print(f"  Section {section_num} ({SECTION_NAMES[int(section_num)]}): {len(questions)} questions")
    
    # Store in database or memory
    if store_interview_document(interview_doc.to_dict()):
        print(f"[Setup] Created 4-section interview session with code: {access_code}")
        
        return SetupInterviewOut(
            access_code=access_code,
            message=f"4-section interview session created successfully. Use access code {access_code} to start your interview."
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to create interview session")

@router.post("/start-interview", response_model=StartInterviewOut)
def start_interview(inp: StartInterviewIn):
    """Starts the interview using the 5-digit access code"""
    print(f"[Start] Starting 4-section interview with code: {inp.access_code}")
    
    # Get interview session
    interview_doc = get_interview_document(inp.access_code)
    if not interview_doc:
        raise HTTPException(status_code=404, detail="Invalid access code")
    
    # Get first question from section 1
    question_text, question_number, section, section_name = get_current_question_info(interview_doc)
    
    print(f"[Start] Starting Section {section} ({section_name}): {question_text}")
    
    # Convert to speech
    question_audio = text_to_speech(question_text)
    
    total_questions = sum(SECTION_QUESTION_COUNTS.values())
    
    return StartInterviewOut(
        access_code=inp.access_code,
        question_text=question_text,
        question_audio=question_audio,
        question_number=question_number,
        section=section,
        section_name=section_name,
        total_questions=total_questions
    )

@router.post("/answer-question", response_model=AnswerQuestionOut)
def answer_question(inp: AnswerQuestionIn):
    """Processes candidate's audio answer and returns next question"""
    print(f"[Answer] Processing answer for code: {inp.access_code}")
    
    # Get interview session
    interview_doc = get_interview_document(inp.access_code)
    if not interview_doc:
        raise HTTPException(status_code=404, detail="Invalid access code")
    
    # Transcribe audio answer
    candidate_answer = transcribe_audio(inp.audio_data)
    print(f"[Answer] Transcribed: {candidate_answer[:100]}...")
    
    # Calculate voice metrics
    audio_bytes = base64.b64decode(inp.audio_data)
    voice_metrics = calculate_voice_metrics(candidate_answer, audio_bytes)
    
    # Get current question details BEFORE advancing
    current_question, question_number, section, section_name = get_current_question_info(interview_doc)
    
    # Create QA pair with voice metrics and section info
    qa_pair = {
        "question": current_question,
        "answer": candidate_answer,
        "question_number": question_number,
        "section": section,
        "voice_metrics": voice_metrics,
        "timestamp": datetime.now()
    }
    
    # Ensure data is serializable for MongoDB
    serializable_qa_pair = ensure_serializable_qa_pair(qa_pair)
    
    print(f"[Answer] Storing QA pair for Section {section}, Question {question_number}")
    
    # Save QA pair with error checking
    success = update_interview_document(
        inp.access_code,
        {"$push": {"qa_pairs": serializable_qa_pair}}
    )
    
    if success:
        print(f"[Answer] Successfully stored QA pair")
        # Verify it was stored
        updated_doc = get_interview_document(inp.access_code)
        qa_count = len(updated_doc.get("qa_pairs", []))
        print(f"[Answer] Total QA pairs now: {qa_count}")
    else:
        print(f"[Answer] WARNING: Failed to store QA pair - continuing interview")
    
    # Advance to next question
    has_more = advance_to_next_question(inp.access_code, interview_doc)
    
    if not has_more:  # Interview complete
        print(f"[Answer] All 4 sections completed for code: {inp.access_code}")
        return AnswerQuestionOut(interview_completed=True)
    
    # Get updated interview document and next question
    updated_interview_doc = get_interview_document(inp.access_code)
    next_question, next_q_number, next_section, next_section_name = get_current_question_info(updated_interview_doc)
    
    print(f"[Answer] Next question - Section {next_section} ({next_section_name}): {next_question}")
    
    # Convert to speech
    question_audio = text_to_speech(next_question)
    
    total_questions = sum(SECTION_QUESTION_COUNTS.values())
    
    return AnswerQuestionOut(
        question_text=next_question,
        question_audio=question_audio,
        question_number=next_q_number,
        section=next_section,
        section_name=next_section_name,
        total_questions=total_questions,
        interview_completed=False
    )

@router.post("/get-evaluation", response_model=GetEvaluationOut)
def get_evaluation(inp: GetEvaluationIn):
    """Returns the evaluation for a specific section (2, 3, or 4 only)"""
    if inp.section not in [2, 3, 4]:
        raise HTTPException(status_code=400, detail="Only sections 2, 3, and 4 can be evaluated")
    
    print(f"[Evaluation] Getting evaluation for code: {inp.access_code}, section: {inp.section}")
    
    # Get interview session
    interview_doc = get_interview_document(inp.access_code)
    if not interview_doc:
        raise HTTPException(status_code=404, detail="Invalid access code")
    
    # Get Q&A pairs for the specific section
    section_qa_pairs = get_section_qa_pairs(interview_doc, inp.section)
    
    if not section_qa_pairs:
        raise HTTPException(status_code=400, detail=f"No Q&A pairs found for section {inp.section}")
    
    # Get evaluation from Gemini for this section
    evaluation = gemini_client.evaluate_section(
        interview_doc["job_role"], 
        inp.section, 
        section_qa_pairs
    )
    
    print(f"[Evaluation] Section {inp.section} evaluation complete. Score: {evaluation['overall_score']}/10")
    
    # Format transcript for response
    transcript = [
        {
            "question": qa["question"],
            "answer": qa["answer"],
            "question_number": str(qa["question_number"])
        }
        for qa in section_qa_pairs
    ]
    
    return GetEvaluationOut(
        section=inp.section,
        section_name=SECTION_NAMES[inp.section],
        overall_score=evaluation["overall_score"],
        feedback=evaluation["feedback"],
        transcript=transcript
    )

@router.post("/generate-report", response_model=GenerateReportOut)
def generate_report(inp: GenerateReportIn):
    """Generates a comprehensive interview report with confidence scoring"""
    print(f"[Report] Generating report for code: {inp.access_code}")
    
    # Get interview session
    interview_doc = get_interview_document(inp.access_code)
    if not interview_doc:
        raise HTTPException(status_code=404, detail="Invalid access code")
    
    if not interview_doc.get("interview_complete", False):
        raise HTTPException(status_code=400, detail="Interview not yet completed")
    
    qa_pairs = interview_doc.get("qa_pairs", [])
    if not qa_pairs:
        raise HTTPException(status_code=400, detail="No interview data found")
    
    # Extract voice metrics
    voice_metrics = [qa["voice_metrics"] for qa in qa_pairs if "voice_metrics" in qa]
    
    # Generate comprehensive report
    report = generate_confidence_report(qa_pairs, voice_metrics)
    
    # Calculate QA summary
    qa_summary = {
        "total_questions": len(qa_pairs),
        "avg_answer_length": np.mean([len(qa["answer"].split()) for qa in qa_pairs]),
        "total_interview_duration": "Estimated based on questions answered"
    }
    
    print(f"[Report] Generated report with confidence score: {report['confidence_score']:.3f}")
    
    return GenerateReportOut(
        access_code=inp.access_code,
        confidence_score=report["confidence_score"],
        detailed_analysis=report["detailed_analysis"],
        voice_metrics_summary=report["voice_metrics_summary"],
        qa_summary=qa_summary
    )

@router.get("/debug/{access_code}")
def debug_interview_status(access_code: str):
    """Debug endpoint to check interview data storage"""
    interview_doc = get_interview_document(access_code)
    if not interview_doc:
        raise HTTPException(status_code=404, detail="Invalid access code")
    
    qa_pairs = interview_doc.get("qa_pairs", [])
    
    debug_info = {
        "access_code": access_code,
        "storage_mode": "MongoDB" if mongodb_available else "Memory",
        "total_qa_pairs": len(qa_pairs),
        "interview_complete": interview_doc.get("interview_complete", False),
        "current_section": interview_doc.get("current_section", 1),
        "current_question_in_section": interview_doc.get("current_question_in_section", 0),
        "qa_pairs_by_section": {},
        "voice_metrics_sample": None
    }
    
    # Group QA pairs by section
    for section in range(1, 5):
        section_qa = [qa for qa in qa_pairs if qa.get("section") == section]
        debug_info["qa_pairs_by_section"][section] = {
            "count": len(section_qa),
            "questions": [qa.get("question", "")[:50] + "..." for qa in section_qa]
        }
    
    # Sample voice metrics from first QA pair
    if qa_pairs:
        debug_info["voice_metrics_sample"] = qa_pairs[0].get("voice_metrics", {})
    
    return debug_info

@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "interview-chatbot-mongo",
        "mongodb_connected": mongodb_available,
        "storage_mode": "MongoDB" if mongodb_available else "In-Memory"
    }

@router.get("/debug/qa-storage/{access_code}")
def debug_qa_storage(access_code: str):
    """Debug Q&A storage issues"""
    doc = get_interview_document(access_code)
    if not doc:
        return {"error": "Document not found"}
    
    return {
        "qa_pairs_count": len(doc.get("qa_pairs", [])),
        "qa_pairs_preview": doc.get("qa_pairs", [])[:2],  # First 2 QA pairs
        "current_section": doc.get("current_section"),
        "interview_complete": doc.get("interview_complete"),
        "section_questions_counts": {
            section: len(questions) 
            for section, questions in doc.get("section_questions", {}).items()
        }
    }

@router.get("/session/{access_code}")
def get_session_status(access_code: str):
    """Get interview session status with section progress"""
    interview_doc = get_interview_document(access_code)
    if not interview_doc:
        raise HTTPException(status_code=404, detail="Invalid access code")
    
    total_questions = sum(SECTION_QUESTION_COUNTS.values())
    answered_questions = len(interview_doc.get("qa_pairs", []))
    
    # Calculate sections completed
    sections_completed = []
    for section in range(1, 5):
        section_qa_count = len([qa for qa in interview_doc.get("qa_pairs", []) if qa.get("section") == section])
        if section_qa_count == SECTION_QUESTION_COUNTS[section]:
            sections_completed.append(section)
    
    return {
        "access_code": access_code,
        "job_role": interview_doc["job_role"],
        "current_section": interview_doc.get("current_section", 1),
        "current_section_name": SECTION_NAMES[interview_doc.get("current_section", 1)],
        "progress": {
            "answered_questions": answered_questions,
            "total_questions": total_questions,
            "percentage": round((answered_questions / total_questions) * 100, 1)
        },
        "questions_answered": answered_questions,
        "interview_complete": interview_doc.get("interview_complete", False),
        "sections_completed": sections_completed,
        "created_at": interview_doc["created_at"],
        "storage_mode": "MongoDB" if mongodb_available else "In-Memory"
    }