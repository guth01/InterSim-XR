from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import base64
import io
import json
import uuid
from datetime import datetime
from openai import OpenAI
import google.generativeai as genai

# Initialize clients
openai_client = OpenAI()  # Make sure OPENAI_API_KEY is set
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

router = APIRouter(prefix="/interview", tags=["interview"])

# ===== DTOs =====

class StartInterviewIn(BaseModel):
    job_role: str = "Data Scientist"
    job_description: Optional[str] = None

class StartInterviewOut(BaseModel):
    session_id: str
    question_text: str
    question_audio: str  # base64 encoded audio
    question_number: int

class AnswerQuestionIn(BaseModel):
    session_id: str
    audio_data: str  # base64 encoded WAV

class AnswerQuestionOut(BaseModel):
    question_text: Optional[str] = None  # Next question, None if interview complete
    question_audio: Optional[str] = None  # base64 encoded audio
    question_number: Optional[int] = None
    interview_completed: bool = False

class GetEvaluationIn(BaseModel):
    session_id: str

class GetEvaluationOut(BaseModel):
    overall_score: int  # 1-10
    feedback: str
    transcript: List[Dict[str, str]]  # List of Q&A pairs

class QuestionAnswer(BaseModel):
    question: str
    answer: str
    question_number: int
    timestamp: datetime

# ===== Managers and Clients =====

class InterviewManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
    
    def create_session(self, job_role: str, job_description: Optional[str] = None) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "job_role": job_role,
            "job_description": job_description,
            "questions": [],
            "answers": [],
            "qa_pairs": [],
            "current_question_index": 0,
            "interview_complete": False,
            "needs_follow_up": False,
            "last_answer_weak": False,
            "created_at": datetime.now()
        }
        return session_id
    
    def get_session(self, session_id: str) -> Dict:
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return self.sessions[session_id]
    
    def add_qa_pair(self, session_id: str, question: str, answer: str, question_number: int):
        session = self.get_session(session_id)
        qa_pair = {
            "question": question,
            "answer": answer,
            "question_number": question_number,
            "timestamp": datetime.now().isoformat()
        }
        session["qa_pairs"].append(qa_pair)
        
        # Store in placeholder DB
        db_manager.store_qa_pair(session_id, qa_pair)
    
    def is_interview_complete(self, session_id: str) -> bool:
        session = self.get_session(session_id)
        # Complete if we've asked 5-7 main questions (excluding follow-ups)
        main_questions = len([qa for qa in session["qa_pairs"] if not qa.get("is_follow_up", False)])
        return main_questions >= 5 and not session.get("needs_follow_up", False)
    
    def mark_complete(self, session_id: str):
        session = self.get_session(session_id)
        session["interview_complete"] = True

class DBManager:
    def __init__(self):
        self.qa_storage = {}  # In-memory storage - replace with actual DB
    
    def store_qa_pair(self, session_id: str, qa_pair: Dict):
        if session_id not in self.qa_storage:
            self.qa_storage[session_id] = []
        self.qa_storage[session_id].append(qa_pair)
        print(f"[DB] Stored Q&A for session {session_id}: Q='{qa_pair['question'][:50]}...' A='{qa_pair['answer'][:50]}...'")
    
    def get_qa_pairs(self, session_id: str) -> List[Dict]:
        return self.qa_storage.get(session_id, [])

class GeminiClient:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def evaluate_interview(self, job_role: str, qa_pairs: List[Dict]) -> Dict[str, Any]:
        # Prepare transcript for evaluation
        transcript = ""
        for i, qa in enumerate(qa_pairs, 1):
            transcript += f"Q{i}: {qa['question']}\nA{i}: {qa['answer']}\n\n"
        
        evaluation_prompt = f"""
        You are an expert interviewer evaluating a candidate for a {job_role} position.
        
        Interview Transcript:
        {transcript}
        
        Please provide:
        1. An overall score from 1-10 (where 10 is excellent)
        2. Detailed feedback on the candidate's performance
        
        Consider:
        - Technical knowledge and accuracy
        - Communication skills
        - Problem-solving approach
        - Depth of understanding
        - Areas for improvement
        
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
                "overall_score": max(1, min(10, score)),  # Ensure 1-10 range
                "feedback": feedback
            }
        except Exception as e:
            print(f"[Gemini] Evaluation failed: {e}")
            # Return fallback evaluation
            return {
                "overall_score": 7,
                "feedback": f"Interview completed successfully. {len(qa_pairs)} questions answered. Due to evaluation service unavailability, a detailed analysis cannot be provided at this time."
            }

# Initialize managers
interview_manager = InterviewManager()
db_manager = DBManager()
gemini_client = GeminiClient()

# ===== Helper Functions =====

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

def generate_initial_questions(job_role: str, job_description: Optional[str] = None) -> List[str]:
    """Generates initial interview questions using OpenAI"""
    description_context = f"\nJob Description: {job_description}" if job_description else ""
    
    system_prompt = f"""You are an expert interviewer conducting a technical interview for a {job_role} role.{description_context}

Generate exactly 6 interview questions that:
1. Start with a warm introduction/icebreaker
2. Cover technical skills relevant to the role
3. Include problem-solving scenarios
4. Assess experience and past projects
5. Explore knowledge depth
6. End with questions about challenges/growth

Return ONLY the questions, numbered 1-6, one per line."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate interview questions for {job_role}"}
            ],
            temperature=0.7
        )
        
        questions_text = response.choices[0].message.content
        questions = [q.strip() for q in questions_text.split('\n') if q.strip() and any(c.isdigit() for c in q[:3])]
        
        # Clean up numbering
        cleaned_questions = []
        for q in questions:
            # Remove numbering prefix (1., 2., etc.)
            clean_q = q.split('.', 1)[-1].strip() if '.' in q[:5] else q
            cleaned_questions.append(clean_q)
        
        return cleaned_questions[:6]  # Ensure exactly 6 questions
        
    except Exception as e:
        print(f"[OpenAI] Question generation error: {e}")
        # Fallback questions
        return [
            f"Could you start by telling me about yourself and what interests you about the {job_role} role?",
            "Can you walk me through a challenging technical project you've worked on recently?",
            "How do you approach problem-solving when faced with a technical issue you haven't encountered before?",
            "What tools and technologies are you most comfortable working with, and why?",
            "Can you describe a time when you had to learn a new technology or skill quickly?",
            "What do you see as the biggest challenges in the field right now, and how do you stay current?"
        ]

def generate_next_question(session: Dict, candidate_answer: str) -> tuple[str, bool]:
    """Generates the next conversational question based on previous answers"""
    qa_pairs = session["qa_pairs"]
    current_index = len([qa for qa in qa_pairs if not qa.get("is_follow_up", False)])
    
    # Get initial questions if not generated yet
    if "initial_questions" not in session:
        session["initial_questions"] = generate_initial_questions(session["job_role"], session.get("job_description"))
    
    initial_questions = session["initial_questions"]
    
    # Check if we need a follow-up question
    if assess_answer_quality(candidate_answer):
        print(f"[Interview] Answer seems weak, generating follow-up")
        follow_up = generate_follow_up_question(qa_pairs[-1]["question"], candidate_answer)
        return follow_up, True
    
    # Check if interview should be complete
    if current_index >= len(initial_questions):
        return "", False  # Interview complete
    
    # Generate next main question
    next_base_question = initial_questions[current_index]
    
    # Make it conversational by referencing previous answer
    if qa_pairs:
        conversational_question = make_question_conversational(next_base_question, candidate_answer)
        return conversational_question, False
    
    return next_base_question, False

def assess_answer_quality(answer: str) -> bool:
    """Simple heuristic to determine if answer needs follow-up"""
    if not answer or len(answer.strip()) < 20:
        return True
    
    # Check for vague responses
    vague_indicators = ["i don't know", "not sure", "maybe", "i think", "probably"]
    answer_lower = answer.lower()
    vague_count = sum(1 for indicator in vague_indicators if indicator in answer_lower)
    
    return vague_count > 0 and len(answer.split()) < 30

def generate_follow_up_question(original_question: str, weak_answer: str) -> str:
    """Generates a probing follow-up question"""
    prompt = f"""The candidate was asked: "{original_question}"
Their response was: "{weak_answer}"

Generate a gentle but probing follow-up question that helps them elaborate or provide a more specific example. Keep it conversational and encouraging."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a supportive interviewer helping candidates give their best answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except:
        return "Could you elaborate a bit more on that? Perhaps with a specific example?"

def make_question_conversational(base_question: str, previous_answer: str) -> str:
    """Makes the next question conversational by referencing the previous answer"""
    prompt = f"""The candidate just answered: "{previous_answer[:200]}..."

Now I need to ask: "{base_question}"

Make this transition natural and conversational by:
1. Briefly acknowledging their previous answer (1 sentence)
2. Smoothly transitioning to the new question

Keep it warm and professional."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a skilled interviewer who makes conversations flow naturally."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except:
        # Fallback to simple transition
        return f"That's helpful to know. {base_question}"

# ===== API Endpoints =====

@router.post("/start-interview", response_model=StartInterviewOut)
def start_interview(inp: StartInterviewIn):
    """Starts a new interview session and asks the first question"""
    print(f"[Interview] Starting new interview for role: {inp.job_role}")
    
    # Create session
    session_id = interview_manager.create_session(inp.job_role, inp.job_description)
    session = interview_manager.get_session(session_id)
    
    # Generate initial questions
    session["initial_questions"] = generate_initial_questions(inp.job_role, inp.job_description)
    first_question = session["initial_questions"][0]
    
    print(f"[Interview] Generated first question: {first_question}")
    
    # Convert to speech
    question_audio = text_to_speech(first_question)
    
    return StartInterviewOut(
        session_id=session_id,
        question_text=first_question,
        question_audio=question_audio,
        question_number=1
    )

@router.post("/answer-question", response_model=AnswerQuestionOut)
def answer_question(inp: AnswerQuestionIn):
    """Processes candidate's audio answer and returns next question"""
    print(f"[Interview] Processing answer for session: {inp.session_id}")
    
    session = interview_manager.get_session(inp.session_id)
    
    # Transcribe audio answer
    candidate_answer = transcribe_audio(inp.audio_data)
    print(f"[Interview] Transcribed answer: {candidate_answer[:100]}...")
    
    # Get the current question (last question asked)
    qa_pairs = session["qa_pairs"]
    if not qa_pairs:
        # This is the answer to the first question
        current_question = session["initial_questions"][0]
        question_number = 1
    else:
        # Get the last question asked
        last_qa = qa_pairs[-1]
        if session.get("waiting_for_follow_up"):
            current_question = session["last_follow_up_question"]
            question_number = last_qa["question_number"]
        else:
            # This shouldn't happen, but handle gracefully
            current_question = "Previous question"
            question_number = len(qa_pairs) + 1
    
    # Save Q&A pair
    interview_manager.add_qa_pair(inp.session_id, current_question, candidate_answer, question_number)
    
    # Generate next question
    next_question, is_follow_up = generate_next_question(session, candidate_answer)
    
    if not next_question:  # Interview complete
        print(f"[Interview] Interview completed for session: {inp.session_id}")
        interview_manager.mark_complete(inp.session_id)
        return AnswerQuestionOut(interview_completed=True)
    
    print(f"[Interview] Next question ({'follow-up' if is_follow_up else 'main'}): {next_question}")
    
    # Update session state
    if is_follow_up:
        session["waiting_for_follow_up"] = True
        session["last_follow_up_question"] = next_question
        next_question_number = question_number  # Same number for follow-up
    else:
        session["waiting_for_follow_up"] = False
        next_question_number = len([qa for qa in qa_pairs if not qa.get("is_follow_up", False)]) + 1
    
    # Convert to speech
    question_audio = text_to_speech(next_question)
    
    return AnswerQuestionOut(
        question_text=next_question,
        question_audio=question_audio,
        question_number=next_question_number,
        interview_completed=False
    )

@router.post("/get-evaluation", response_model=GetEvaluationOut)
def get_evaluation(inp: GetEvaluationIn):
    """Returns the interview evaluation from Gemini"""
    print(f"[Interview] Getting evaluation for session: {inp.session_id}")
    
    session = interview_manager.get_session(inp.session_id)
    
    if not session.get("interview_complete", False):
        raise HTTPException(status_code=400, detail="Interview not yet completed")
    
    qa_pairs = session["qa_pairs"]
    if not qa_pairs:
        raise HTTPException(status_code=400, detail="No Q&A pairs found for evaluation")
    
    # Get evaluation from Gemini
    evaluation = gemini_client.evaluate_interview(session["job_role"], qa_pairs)
    
    print(f"[Interview] Evaluation complete. Score: {evaluation['overall_score']}/10")
    
    # Format transcript for response
    transcript = [
        {
            "question": qa["question"],
            "answer": qa["answer"],
            "question_number": str(qa["question_number"])
        }
        for qa in qa_pairs
    ]
    
    return GetEvaluationOut(
        overall_score=evaluation["overall_score"],
        feedback=evaluation["feedback"],
        transcript=transcript
    )

# Optional: Health check endpoint
@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "interview-chatbot"}