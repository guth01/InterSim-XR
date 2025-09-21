#!/usr/bin/env python3
"""
Test script for Interview Chatbot API endpoints
Tests the complete interview flow with real TTS/STT functionality
"""

import requests
import base64
import json
import time
import wave
import io
from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
BASE_URL = "http://localhost:8000"
INTERVIEW_URL = f"{BASE_URL}/interview"

class AudioTester:
    """Handles audio generation and processing for testing"""
    
    def __init__(self):
        # Check if required environment variables are set
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Import OpenAI client for generating test audio
        from openai import OpenAI
        self.openai_client = OpenAI()
    
    def create_test_audio(self, text: str) -> str:
        """
        Creates test audio from text using OpenAI TTS
        Returns base64 encoded WAV audio
        """
        try:
            print(f"🎵 Generating test audio for: '{text[:50]}...'")
            
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice="alloy",  # Different voice than the interviewer
                input=text,
                response_format="wav"
            )
            
            audio_bytes = response.content
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            print(f"✅ Generated {len(audio_bytes)} bytes of audio")
            return audio_base64
            
        except Exception as e:
            print(f"❌ Failed to generate test audio: {e}")
            raise

class InterviewTester:
    """Main test class for interview API endpoints"""
    
    def __init__(self):
        self.audio_tester = AudioTester()
        self.session_id = None
        self.question_count = 0
        
        # Predefined candidate answers for testing
        self.test_answers = [
            "Hi! I'm a passionate data scientist with 3 years of experience in machine learning and analytics. I'm excited about this role because it combines my love for data with the opportunity to work on impactful projects.",
            
            "Recently, I worked on a customer churn prediction model for an e-commerce company. The challenge was dealing with imbalanced data and feature engineering. I used SMOTE for balancing and created interaction features that improved our F1 score from 0.65 to 0.82.",
            
            "When I encounter a new technical problem, I first break it down into smaller components. I research similar solutions, consult documentation, and often prototype different approaches. I also reach out to colleagues or online communities when needed.",
            
            "I'm most comfortable with Python, especially pandas, scikit-learn, and TensorFlow. I also use SQL for data extraction and Tableau for visualization. I prefer Python because of its extensive ecosystem and readability.",
            
            "Last year, I had to quickly learn Apache Spark for a big data project. I spent weekends going through tutorials, built small practice projects, and paired with a senior engineer. Within two weeks, I was contributing effectively to the team.",
            
            "I think the biggest challenge is the rapid pace of AI advancement and ensuring ethical AI practices. I stay current by following research papers, attending webinars, and participating in Kaggle competitions."
        ]
        
        self.current_answer_index = 0
    
    def get_next_test_answer(self) -> str:
        """Returns the next predefined test answer"""
        if self.current_answer_index < len(self.test_answers):
            answer = self.test_answers[self.current_answer_index]
            self.current_answer_index += 1
            return answer
        else:
            # Fallback answers if we run out
            return f"This is my response to question {self.current_answer_index + 1}. I believe my experience and skills make me a good fit for this position."
    
    def test_health_endpoint(self) -> bool:
        """Test the health check endpoint"""
        print("\n🔍 Testing health endpoint...")
        
        try:
            response = requests.get(f"{BASE_URL}/health")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_start_interview(self, job_role: str = "Data Scientist", job_description: str = None) -> Dict[str, Any]:
        """Test the start-interview endpoint"""
        print(f"\n🚀 Testing start-interview endpoint for role: {job_role}")
        
        payload = {
            "job_role": job_role
        }
        
        if job_description:
            payload["job_description"] = job_description
        
        try:
            response = requests.post(f"{INTERVIEW_URL}/start-interview", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                self.session_id = data["session_id"]
                self.question_count = data["question_number"]
                
                print(f"✅ Interview started successfully!")
                print(f"   Session ID: {self.session_id}")
                print(f"   Question #{data['question_number']}: {data['question_text']}")
                print(f"   Audio length: {len(data['question_audio'])} characters (base64)")
                
                # Save the first question audio for verification
                self.save_audio_sample(data['question_audio'], f"question_{data['question_number']}.wav")
                
                return data
            else:
                print(f"❌ Start interview failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Start interview error: {e}")
            return None
    
    def test_answer_question(self, answer_text: str) -> Dict[str, Any]:
        """Test the answer-question endpoint with generated audio"""
        print(f"\n🎤 Testing answer-question endpoint...")
        print(f"   Answer: '{answer_text[:100]}...'")
        
        if not self.session_id:
            print("❌ No active session. Call test_start_interview first.")
            return None
        
        try:
            # Generate audio from the answer text
            audio_data = self.audio_tester.create_test_audio(answer_text)
            
            payload = {
                "session_id": self.session_id,
                "audio_data": audio_data
            }
            
            response = requests.post(f"{INTERVIEW_URL}/answer-question", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("interview_completed", False):
                    print("✅ Answer processed - Interview completed!")
                    return data
                else:
                    self.question_count = data["question_number"]
                    print(f"✅ Answer processed successfully!")
                    print(f"   Next Question #{data['question_number']}: {data['question_text']}")
                    print(f"   Audio length: {len(data['question_audio'])} characters (base64)")
                    
                    # Save the question audio
                    self.save_audio_sample(data['question_audio'], f"question_{data['question_number']}.wav")
                    
                return data
            else:
                print(f"❌ Answer question failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Answer question error: {e}")
            return None
    
    def test_get_evaluation(self) -> Dict[str, Any]:
        """Test the get-evaluation endpoint"""
        print(f"\n📊 Testing get-evaluation endpoint...")
        
        if not self.session_id:
            print("❌ No active session. Complete an interview first.")
            return None
        
        try:
            payload = {
                "session_id": self.session_id
            }
            
            response = requests.post(f"{INTERVIEW_URL}/get-evaluation", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ Evaluation retrieved successfully!")
                print(f"   Overall Score: {data['overall_score']}/10")
                print(f"   Feedback: {data['feedback'][:200]}...")
                print(f"   Transcript length: {len(data['transcript'])} Q&A pairs")
                
                return data
            else:
                print(f"❌ Get evaluation failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Get evaluation error: {e}")
            return None
    
    def save_audio_sample(self, audio_base64: str, filename: str):
        """Save base64 audio to file for manual verification"""
        try:
            audio_bytes = base64.b64decode(audio_base64)
            
            # Create audio_samples directory if it doesn't exist
            os.makedirs("audio_samples", exist_ok=True)
            
            filepath = os.path.join("audio_samples", filename)
            with open(filepath, "wb") as f:
                f.write(audio_bytes)
            
            print(f"   💾 Saved audio sample: {filepath}")
            
        except Exception as e:
            print(f"   ⚠️ Failed to save audio sample: {e}")
    
    def run_complete_interview_test(self, job_role: str = "Data Scientist"):
        """Run a complete interview simulation from start to finish"""
        print("=" * 80)
        print("🤖 STARTING COMPLETE INTERVIEW TEST")
        print("=" * 80)
        
        # Test 1: Health check
        if not self.test_health_endpoint():
            print("❌ Health check failed. Stopping test.")
            return False
        
        # Test 2: Start interview
        start_result = self.test_start_interview(job_role)
        if not start_result:
            print("❌ Failed to start interview. Stopping test.")
            return False
        
        # Test 3: Answer questions until interview is complete
        interview_complete = False
        max_questions = 10  # Safety limit
        question_count = 0
        
        while not interview_complete and question_count < max_questions:
            question_count += 1
            
            # Get the next test answer
            answer_text = self.get_next_test_answer()
            
            # Submit the answer
            answer_result = self.test_answer_question(answer_text)
            
            if not answer_result:
                print("❌ Failed to submit answer. Stopping test.")
                return False
            
            interview_complete = answer_result.get("interview_completed", False)
            
            # Small delay to simulate real conversation
            time.sleep(1)
        
        if not interview_complete:
            print("⚠️ Interview didn't complete within expected question limit")
            return False
        
        # Test 4: Get evaluation
        evaluation_result = self.test_get_evaluation()
        if not evaluation_result:
            print("❌ Failed to get evaluation. Test incomplete.")
            return False
        
        print("\n" + "=" * 80)
        print("🎉 COMPLETE INTERVIEW TEST PASSED!")
        print("=" * 80)
        print(f"📊 Final Results:")
        print(f"   Session ID: {self.session_id}")
        print(f"   Questions answered: {question_count}")
        print(f"   Final score: {evaluation_result['overall_score']}/10")
        print(f"   Audio samples saved in: ./audio_samples/")
        
        return True

def main():
    """Main test runner"""
    print("🧪 Interview API Test Suite")
    print("=" * 50)
    
    # Check if the server is running
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Server is running: {response.json()}")
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        print("💡 Make sure to start the server with: python main.py")
        return
    
    # Initialize tester
    try:
        tester = InterviewTester()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("💡 Make sure your .env file contains OPENAI_API_KEY and GEMINI_API_KEY")
        return
    
    # Run the complete test
    success = tester.run_complete_interview_test("Data Scientist")
    
    if success:
        print("\n🎯 All tests passed! The interview API is working correctly.")
    else:
        print("\n💥 Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()