#!/usr/bin/env python3
"""
Simple API test script for Interview Chatbot endpoints
Tests the API logic without real audio processing (uses mock audio data)
"""

import requests
import base64
import json
import time
from typing import Dict, Any

# API Configuration
BASE_URL = "http://localhost:8000"
INTERVIEW_URL = f"{BASE_URL}/interview"

class SimpleAPITester:
    """Simple tester that uses mock audio data"""
    
    def __init__(self):
        self.session_id = None
        self.question_count = 0
        
        # Mock base64 audio data (small WAV file header + minimal data)
        # This is a valid but very short WAV file
        self.mock_audio_data = self.create_mock_audio()
        
        # Test answers for the interview
        self.test_answers = [
            "Hi! I'm a passionate data scientist with 3 years of experience in machine learning and analytics.",
            "I recently worked on a customer churn prediction model that improved our F1 score significantly.",
            "I approach problems by breaking them down, researching solutions, and prototyping different approaches.",
            "I'm comfortable with Python, pandas, scikit-learn, and TensorFlow for data science projects.",
            "I quickly learned Apache Spark for a big data project by studying tutorials and practicing.",
            "The biggest challenge is staying current with AI advancements while ensuring ethical practices."
        ]
        self.current_answer_index = 0
    
    def create_mock_audio(self) -> str:
        """Create a minimal valid WAV file as base64"""
        # Minimal WAV file header (44 bytes) + 1 sample
        wav_header = bytearray([
            0x52, 0x49, 0x46, 0x46,  # "RIFF"
            0x2E, 0x00, 0x00, 0x00,  # File size (46 bytes)
            0x57, 0x41, 0x56, 0x45,  # "WAVE"
            0x66, 0x6D, 0x74, 0x20,  # "fmt "
            0x10, 0x00, 0x00, 0x00,  # Subchunk size (16)
            0x01, 0x00,              # Audio format (PCM)
            0x01, 0x00,              # Channels (1)
            0x44, 0xAC, 0x00, 0x00,  # Sample rate (44100)
            0x88, 0x58, 0x01, 0x00,  # Byte rate
            0x02, 0x00,              # Block align
            0x10, 0x00,              # Bits per sample (16)
            0x64, 0x61, 0x74, 0x61,  # "data"
            0x02, 0x00, 0x00, 0x00,  # Data size (2 bytes)
            0x00, 0x00               # Sample data
        ])
        return base64.b64encode(wav_header).decode('utf-8')
    
    def get_next_test_answer(self) -> str:
        """Get the next test answer"""
        if self.current_answer_index < len(self.test_answers):
            answer = self.test_answers[self.current_answer_index]
            self.current_answer_index += 1
            return answer
        else:
            return f"This is test answer number {self.current_answer_index + 1}."
    
    def test_api_endpoint(self, method: str, endpoint: str, payload: dict = None) -> Dict[str, Any]:
        """Generic API test method"""
        url = f"{INTERVIEW_URL}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url)
            elif method.upper() == "POST":
                response = requests.post(url, json=payload)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else None,
                "error": response.text if response.status_code != 200 else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "status_code": None,
                "data": None,
                "error": str(e)
            }
    
    def test_health(self):
        """Test health endpoint"""
        print("\n🔍 Testing /health endpoint...")
        result = self.test_api_endpoint("GET", "/health")
        
        if result["success"]:
            print(f"✅ Health check passed: {result['data']}")
        else:
            print(f"❌ Health check failed: {result['error']}")
        
        return result["success"]
    
    def test_start_interview(self, job_role: str = "Data Scientist"):
        """Test start interview endpoint"""
        print(f"\n🚀 Testing /start-interview endpoint...")
        
        payload = {"job_role": job_role}
        result = self.test_api_endpoint("POST", "/start-interview", payload)
        
        if result["success"]:
            data = result["data"]
            self.session_id = data["session_id"]
            self.question_count = data["question_number"]
            
            print(f"✅ Interview started successfully!")
            print(f"   Session ID: {self.session_id}")
            print(f"   Question #{data['question_number']}: {data['question_text'][:100]}...")
            print(f"   Has audio data: {len(data['question_audio']) > 0}")
        else:
            print(f"❌ Start interview failed: {result['error']}")
        
        return result["success"]
    
    def test_answer_question(self, answer_text: str):
        """Test answer question endpoint"""
        print(f"\n🎤 Testing /answer-question endpoint...")
        print(f"   Simulating answer: '{answer_text[:50]}...'")
        
        if not self.session_id:
            print("❌ No session ID available")
            return False
        
        payload = {
            "session_id": self.session_id,
            "audio_data": self.mock_audio_data
        }
        
        result = self.test_api_endpoint("POST", "/answer-question", payload)
        
        if result["success"]:
            data = result["data"]
            
            if data.get("interview_completed", False):
                print("✅ Answer processed - Interview completed!")
                return {"success": True, "completed": True}
            else:
                self.question_count = data["question_number"]
                print(f"✅ Answer processed successfully!")
                print(f"   Next Question #{data['question_number']}: {data['question_text'][:100]}...")
                print(f"   Has audio data: {len(data['question_audio']) > 0}")
                return {"success": True, "completed": False}
        else:
            print(f"❌ Answer question failed: {result['error']}")
            return {"success": False, "completed": False}
    
    def test_get_evaluation(self):
        """Test get evaluation endpoint"""
        print(f"\n📊 Testing /get-evaluation endpoint...")
        
        if not self.session_id:
            print("❌ No session ID available")
            return False
        
        payload = {"session_id": self.session_id}
        result = self.test_api_endpoint("POST", "/get-evaluation", payload)
        
        if result["success"]:
            data = result["data"]
            print(f"✅ Evaluation retrieved successfully!")
            print(f"   Overall Score: {data['overall_score']}/10")
            print(f"   Feedback length: {len(data['feedback'])} characters")
            print(f"   Transcript entries: {len(data['transcript'])}")
            print(f"   Sample feedback: {data['feedback'][:150]}...")
        else:
            print(f"❌ Get evaluation failed: {result['error']}")
        
        return result["success"]
    
    def run_full_test_suite(self):
        """Run the complete test suite"""
        print("🧪 Starting Simple API Test Suite")
        print("=" * 60)
        
        # Step 1: Test health
        if not self.test_health():
            return False
        
        # Step 2: Start interview
        if not self.test_start_interview():
            return False
        
        # Step 3: Answer questions until complete
        max_questions = 8
        for i in range(max_questions):
            answer_text = self.get_next_test_answer()
            result = self.test_answer_question(answer_text)
            
            if not result["success"]:
                return False
            
            if result["completed"]:
                print(f"\n🎯 Interview completed after {i + 1} questions")
                break
            
            time.sleep(0.5)  # Small delay between questions
        else:
            print("⚠️ Interview didn't complete within expected question limit")
            return False
        
        # Step 4: Get evaluation
        if not self.test_get_evaluation():
            return False
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        return True

def main():
    """Main test runner"""
    print("🧪 Simple Interview API Test")
    
    # Check server availability
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Server is running: {response.json()['message']}")
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        print("💡 Start the server with: python main.py")
        return
    
    # Run tests
    tester = SimpleAPITester()
    success = tester.run_full_test_suite()
    
    if success:
        print("🎯 All API endpoints are working correctly!")
    else:
        print("💥 Some tests failed. Check the server logs for details.")

if __name__ == "__main__":
    main()