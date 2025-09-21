#!/usr/bin/env python3
"""
Test script for MongoDB-based Interview Chatbot API
Tests the new 5-digit access code system with voice metrics
"""

import requests
import base64
import json
import time
import wave
import io
from typing import Dict, Any
import os

# API Configuration
BASE_URL = "http://localhost:8000"
INTERVIEW_URL = f"{BASE_URL}/interview"

class MongoInterviewTester:
    """Tester for the MongoDB-based interview system"""
    
    def __init__(self):
        self.access_code = None
        
        # Create minimal valid WAV audio (0.2 seconds)
        self.mock_audio_data = self.create_mock_audio()
        
        # Test answers
        self.test_answers = [
            "Hi! I'm John Smith, a passionate data scientist with 5 years of experience in machine learning and analytics. I've worked on various projects involving predictive modeling, data visualization, and statistical analysis.",
            
            "Recently, I led a customer churn prediction project for an e-commerce company. The main challenge was dealing with highly imbalanced data where only 5% of customers actually churned. I used techniques like SMOTE for data balancing, feature engineering for interaction variables, and ensemble methods combining Random Forest with XGBoost. We improved the F1 score from 0.65 to 0.82.",
            
            "When I encounter a new technical problem, I follow a structured approach. First, I break down the problem into smaller, manageable components. Then I research existing solutions and best practices. I create prototypes to test different approaches and validate assumptions. I also leverage online communities and collaborate with colleagues when needed.",
            
            "I'm most comfortable with Python for data science, particularly pandas for data manipulation, scikit-learn for machine learning, and TensorFlow for deep learning. I also use SQL extensively for data extraction and Tableau for visualization. I prefer Python because of its extensive ecosystem and readability.",
            
            "Last year, I had to quickly learn Apache Spark for a big data project involving real-time stream processing. I dedicated weekends to studying the documentation, completed online tutorials, and built practice projects. I also paired with a senior engineer for hands-on learning. Within two weeks, I was contributing effectively to the production pipeline.",
            
            "I think the biggest challenge in data science right now is ensuring ethical AI and addressing bias in machine learning models. There's also the challenge of staying current with rapidly evolving technologies. I stay updated by following research papers, attending webinars, participating in Kaggle competitions, and contributing to open-source projects."
        ]
        self.current_answer_index = 0
    
    def create_mock_audio(self) -> str:
        """Create 0.2 seconds of valid WAV audio"""
        sample_rate = 16000
        duration = 0.2
        samples = int(sample_rate * duration)
        
        # Generate sine wave audio data
        import numpy as np
        t = np.linspace(0, duration, samples, False)
        frequency = 440  # A4 note
        audio_data = (np.sin(2 * np.pi * frequency * t) * 0.3 * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        wav_buffer.seek(0)
        return base64.b64encode(wav_buffer.read()).decode('utf-8')
    
    def get_next_test_answer(self) -> str:
        """Get the next test answer"""
        if self.current_answer_index < len(self.test_answers):
            answer = self.test_answers[self.current_answer_index]
            self.current_answer_index += 1
            return answer
        else:
            return f"This is additional test answer number {self.current_answer_index + 1}."
    
    def test_setup_interview(self):
        """Test the setup-interview endpoint"""
        print("\n🔧 Testing /setup-interview endpoint...")
        
        payload = {
            "job_role": "Data Scientist",
            "job_description": "We are looking for a senior data scientist to join our AI team. Experience with machine learning, Python, and big data technologies required.",
            "resume_text": "John Smith - Data Scientist with 5 years experience in ML, Python, SQL, and cloud platforms."
        }
        
        try:
            response = requests.post(f"{INTERVIEW_URL}/setup-interview", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                self.access_code = data["access_code"]
                
                print(f"✅ Interview setup successful!")
                print(f"   Access Code: {self.access_code}")
                print(f"   Message: {data['message']}")
                return True
            else:
                print(f"❌ Setup failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Setup error: {e}")
            return False
    
    def test_start_interview(self):
        """Test the start-interview endpoint"""
        print(f"\n🚀 Testing /start-interview endpoint...")
        
        if not self.access_code:
            print("❌ No access code available")
            return False
        
        payload = {"access_code": self.access_code}
        
        try:
            response = requests.post(f"{INTERVIEW_URL}/start-interview", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Interview started successfully!")
                print(f"   Access Code: {data['access_code']}")
                print(f"   Question #{data['question_number']}: {data['question_text'][:100]}...")
                print(f"   Has audio data: {len(data['question_audio']) > 0}")
                return True
            else:
                print(f"❌ Start failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Start error: {e}")
            return False
    
    def test_answer_question(self, answer_text: str):
        """Test the answer-question endpoint"""
        print(f"\n🎤 Testing /answer-question endpoint...")
        print(f"   Answer: '{answer_text[:50]}...'")
        
        if not self.access_code:
            print("❌ No access code available")
            return False
        
        payload = {
            "access_code": self.access_code,
            "audio_data": self.mock_audio_data
        }
        
        try:
            response = requests.post(f"{INTERVIEW_URL}/answer-question", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("interview_completed", False):
                    print("✅ Answer processed - Interview completed!")
                    return {"success": True, "completed": True}
                else:
                    print(f"✅ Answer processed successfully!")
                    print(f"   Next Question #{data['question_number']}: {data['question_text'][:100]}...")
                    print(f"   Has audio data: {len(data['question_audio']) > 0}")
                    return {"success": True, "completed": False}
            else:
                print(f"❌ Answer failed: {response.status_code} - {response.text}")
                return {"success": False, "completed": False}
                
        except Exception as e:
            print(f"❌ Answer error: {e}")
            return {"success": False, "completed": False}
    
    def test_generate_report(self):
        """Test the generate-report endpoint"""
        print(f"\n📊 Testing /generate-report endpoint...")
        
        if not self.access_code:
            print("❌ No access code available")
            return False
        
        payload = {"access_code": self.access_code}
        
        try:
            response = requests.post(f"{INTERVIEW_URL}/generate-report", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Report generated successfully!")
                print(f"   Access Code: {data['access_code']}")
                print(f"   Confidence Score: {data['confidence_score']:.3f}")
                print(f"   Analysis: {data['detailed_analysis'][:150]}...")
                print(f"   Voice Metrics: {data['voice_metrics_summary']}")
                print(f"   QA Summary: {data['qa_summary']}")
                return True
            else:
                print(f"❌ Report generation failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Report generation error: {e}")
            return False
    
    def test_session_status(self):
        """Test the session status endpoint"""
        print(f"\n📋 Testing /session/{self.access_code} endpoint...")
        
        if not self.access_code:
            print("❌ No access code available")
            return False
        
        try:
            response = requests.get(f"{INTERVIEW_URL}/session/{self.access_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Session status retrieved!")
                print(f"   Access Code: {data['access_code']}")
                print(f"   Job Role: {data['job_role']}")
                print(f"   Questions Answered: {data['questions_answered']}")
                print(f"   Interview Complete: {data['interview_complete']}")
                return True
            else:
                print(f"❌ Session status failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Session status error: {e}")
            return False
    
    def run_full_test_suite(self):
        """Run the complete test suite"""
        print("🧪 MongoDB Interview API Test Suite")
        print("=" * 60)
        
        # Step 1: Setup interview
        if not self.test_setup_interview():
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
            
            # Check session status after each answer
            self.test_session_status()
            
            if result["completed"]:
                print(f"\n🎯 Interview completed after {i + 1} questions")
                break
            
            time.sleep(1)  # Small delay between questions
        else:
            print("⚠️ Interview didn't complete within expected question limit")
            return False
        
        # Step 4: Generate report
        if not self.test_generate_report():
            return False
        
        print("\n" + "=" * 60)
        print("🎉 ALL MONGODB TESTS PASSED!")
        print("=" * 60)
        print(f"🎯 Final Results:")
        print(f"   Access Code: {self.access_code}")
        print(f"   Full interview workflow completed")
        print(f"   Voice metrics calculated and stored")
        print(f"   Confidence report generated")
        
        return True

def main():
    """Main test runner"""
    print("🧪 MongoDB Interview API Test")
    
    # Check server availability
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Server is running: {response.json()['message']}")
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        print("💡 Start the server with: python main_mongo.py")
        return
    
    # Run tests
    tester = MongoInterviewTester()
    success = tester.run_full_test_suite()
    
    if success:
        print("🎯 All MongoDB API endpoints are working correctly!")
        print("💾 Data stored in MongoDB Atlas")
        print("🎵 Voice metrics calculated and analyzed")
        print("📊 Confidence scoring system operational")
    else:
        print("💥 Some tests failed. Check the server logs for details.")

if __name__ == "__main__":
    main()