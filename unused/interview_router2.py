from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import base64
import io
import json
import uuid
import random
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
    resume_text: Optional[str] = None

class StartInterviewOut(BaseModel):
    session_id: str
    question_text: str
    question_audio: str  # base64 encoded audio
    question_number: int
    section: int
    section_name: str
    total_questions: int

class AnswerQuestionIn(BaseModel):
    session_id: str
    audio_data: str  # base64 encoded WAV

class AnswerQuestionOut(BaseModel):
    question_text: Optional[str] = None  # Next question, None if interview complete
    question_audio: Optional[str] = None  # base64 encoded audio
    question_number: Optional[int] = None
    section: Optional[int] = None
    section_name: Optional[str] = None
    total_questions: Optional[int] = None
    interview_completed: bool = False

class GetEvaluationIn(BaseModel):
    session_id: str
    section: int  # 2, 3, or 4 (section 1 not evaluated)

class GetEvaluationOut(BaseModel):
    section: int
    section_name: str
    overall_score: int  # 1-10
    feedback: str
    transcript: List[Dict[str, str]]  # List of Q&A pairs for that section

class QuestionAnswer(BaseModel):
    question: str
    answer: str
    question_number: int
    section: int
    timestamp: datetime

# ===== Constants =====

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

# ===== Managers and Clients =====

class InterviewManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
    
    def create_session(self, job_role: str, job_description: Optional[str] = None, 
                      resume_text: Optional[str] = None) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "job_role": job_role,
            "job_description": job_description,
            "resume_text": resume_text,
            "current_section": 1,
            "current_question_in_section": 0,
            "qa_pairs": [],
            "section_questions": {1: [], 2: [], 3: [], 4: []},  # Store questions by section
            "interview_complete": False,
            "created_at": datetime.now()
        }
        
        # Generate all questions for all sections at the start
        self._generate_all_questions(session_id)
        
        return session_id
    
    def _generate_all_questions(self, session_id: str):
        """Generate questions for all sections at session creation"""
        session = self.sessions[session_id]
        
        # Section 1: General questions
        session["section_questions"][1] = self._generate_general_questions()
        
        # Section 2: Role-based questions
        session["section_questions"][2] = self._generate_role_questions(
            session["job_role"], session["job_description"]
        )
        
        # Section 3: Resume-based questions
        session["section_questions"][3] = self._generate_resume_questions(
            session["job_role"], session["resume_text"]
        )
        
        # Section 4: Behavioral questions
        session["section_questions"][4] = self._select_behavioral_questions()
    
    def _generate_general_questions(self) -> List[str]:
        """Generate 2 general introduction questions"""
        return [
            "Could you please introduce yourself and tell me a bit about your background?",
            "What motivated you to apply for this position and what interests you about our company?"
        ]
    
    def _generate_role_questions(self, job_role: str, job_description: Optional[str] = None) -> List[str]:
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
                model="gpt-4",
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
    
    def _generate_resume_questions(self, job_role: str, resume_text: Optional[str] = None) -> List[str]:
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
                model="gpt-4",
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
    
    def _select_behavioral_questions(self) -> List[str]:
        """Select 3 random behavioral questions from the bank"""
        return random.sample(BEHAVIORAL_QUESTIONS, 3)
    
    def get_session(self, session_id: str) -> Dict:
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return self.sessions[session_id]
    
    def get_current_question(self, session_id: str) -> tuple[str, int, int, str]:
        """Returns (question_text, question_number, section, section_name)"""
        session = self.get_session(session_id)
        current_section = session["current_section"]
        current_q_index = session["current_question_in_section"]
        
        section_questions = session["section_questions"][current_section]
        
        if current_q_index >= len(section_questions):
            raise HTTPException(status_code=500, detail="No more questions in current section")
        
        question_text = section_questions[current_q_index]
        question_number = current_q_index + 1
        section_name = SECTION_NAMES[current_section]
        
        return question_text, question_number, current_section, section_name
    
    def advance_to_next_question(self, session_id: str) -> bool:
        """Advances to next question. Returns True if there are more questions, False if interview complete"""
        session = self.get_session(session_id)
        current_section = session["current_section"]
        current_q_index = session["current_question_in_section"]
        
        # Move to next question in current section
        session["current_question_in_section"] += 1
        
        # Check if we've finished current section
        section_questions = session["section_questions"][current_section]
        if session["current_question_in_section"] >= len(section_questions):
            # Move to next section
            if current_section < 4:
                session["current_section"] += 1
                session["current_question_in_section"] = 0
                print(f"[Interview] Advanced to section {session['current_section']}")
                return True
            else:
                # Interview complete
                session["interview_complete"] = True
                print(f"[Interview] Interview completed")
                return False
        
        return True
    
    def add_qa_pair(self, session_id: str, question: str, answer: str, 
                    question_number: int, section: int):
        session = self.get_session(session_id)
        qa_pair = {
            "question": question,
            "answer": answer,
            "question_number": question_number,
            "section": section,
            "timestamp": datetime.now().isoformat()
        }
        session["qa_pairs"].append(qa_pair)
        
        # Store in placeholder DB
        db_manager.store_qa_pair(session_id, qa_pair)
    
    def get_section_qa_pairs(self, session_id: str, section: int) -> List[Dict]:
        """Get Q&A pairs for a specific section"""
        session = self.get_session(session_id)
        return [qa for qa in session["qa_pairs"] if qa["section"] == section]

class DBManager:
    def __init__(self):
        self.qa_storage = {}  # In-memory storage - replace with actual DB
    
    def store_qa_pair(self, session_id: str, qa_pair: Dict):
        if session_id not in self.qa_storage:
            self.qa_storage[session_id] = []
        self.qa_storage[session_id].append(qa_pair)
        print(f"[DB] Stored Q&A for session {session_id}, section {qa_pair['section']}: Q='{qa_pair['question'][:50]}...'")
    
    def get_qa_pairs(self, session_id: str, section: Optional[int] = None) -> List[Dict]:
        qa_pairs = self.qa_storage.get(session_id, [])
        if section:
            return [qa for qa in qa_pairs if qa["section"] == section]
        return qa_pairs

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

# ===== API Endpoints =====

@router.post("/start-interview", response_model=StartInterviewOut)
def start_interview(inp: StartInterviewIn):
    """Starts a new multi-section interview session"""
    print(f"[Interview] Starting new multi-section interview for role: {inp.job_role}")
    
    # Create session with all sections
    session_id = interview_manager.create_session(
        inp.job_role, 
        inp.job_description, 
        inp.resume_text
    )
    
    # Get first question from section 1
    question_text, question_number, section, section_name = interview_manager.get_current_question(session_id)
    
    print(f"[Interview] Starting Section {section} ({section_name}): {question_text}")
    
    # Convert to speech
    question_audio = text_to_speech(question_text)
    
    total_questions = sum(SECTION_QUESTION_COUNTS.values())
    
    return StartInterviewOut(
        session_id=session_id,
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
    print(f"[Interview] Processing answer for session: {inp.session_id}")
    
    session = interview_manager.get_session(inp.session_id)
    
    # Transcribe audio answer
    candidate_answer = transcribe_audio(inp.audio_data)
    print(f"[Interview] Transcribed answer: {candidate_answer[:100]}...")
    
    # Get current question details
    current_question, question_number, section, section_name = interview_manager.get_current_question(inp.session_id)
    
    # Save current Q&A pair
    interview_manager.add_qa_pair(
        inp.session_id, 
        current_question, 
        candidate_answer, 
        question_number, 
        section
    )
    
    # Advance to next question
    has_more = interview_manager.advance_to_next_question(inp.session_id)
    
    if not has_more:  # Interview complete
        print(f"[Interview] All sections completed for session: {inp.session_id}")
        return AnswerQuestionOut(interview_completed=True)
    
    # Get next question
    next_question, next_q_number, next_section, next_section_name = interview_manager.get_current_question(inp.session_id)
    
    print(f"[Interview] Next question - Section {next_section} ({next_section_name}): {next_question}")
    
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
    
    print(f"[Interview] Getting evaluation for session: {inp.session_id}, section: {inp.section}")
    
    session = interview_manager.get_session(inp.session_id)
    
    # Get Q&A pairs for the specific section
    section_qa_pairs = interview_manager.get_section_qa_pairs(inp.session_id, inp.section)
    
    if not section_qa_pairs:
        raise HTTPException(status_code=400, detail=f"No Q&A pairs found for section {inp.section}")
    
    # Get evaluation from Gemini for this section
    evaluation = gemini_client.evaluate_section(
        session["job_role"], 
        inp.section, 
        section_qa_pairs
    )
    
    print(f"[Interview] Section {inp.section} evaluation complete. Score: {evaluation['overall_score']}/10")
    
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

# Optional: Health check endpoint
@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "multi-section-interview-chatbot"}

# Optional: Get session status endpoint
@router.get("/session/{session_id}")
def get_session_status(session_id: str):
    """Get current session status and progress"""
    session = interview_manager.get_session(session_id)
    
    total_questions = sum(SECTION_QUESTION_COUNTS.values())
    answered_questions = len(session["qa_pairs"])
    
    return {
        "session_id": session_id,
        "current_section": session["current_section"],
        "current_section_name": SECTION_NAMES[session["current_section"]],
        "progress": {
            "answered_questions": answered_questions,
            "total_questions": total_questions,
            "percentage": round((answered_questions / total_questions) * 100, 1)
        },
        "interview_complete": session["interview_complete"],
        "sections_completed": [
            section for section in range(1, 5) 
            if len([qa for qa in session["qa_pairs"] if qa["section"] == section]) == SECTION_QUESTION_COUNTS[section]
        ]
    }