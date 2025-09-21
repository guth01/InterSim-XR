#!/usr/bin/env python3
"""
Quick test for the NEW MongoDB-based Interview API
Tests the 5-digit access code system
"""

import requests
import base64
import json
import time
import wave
import io
from typing import Dict, Any

# API Configuration
BASE_URL = "http://localhost:8000"
INTERVIEW_URL = f"{BASE_URL}/interview"

class QuickMongoTest:
    """Quick tester for the new MongoDB API"""
    
    def __init__(self):
        self.access_code = None
        self.mock_audio_data = self.create_mock_audio()
        
        self.test_answers = [
            "Hi! I'm John Smith, a data scientist with 5 years of experience in machine learning.",
            "I recently built a customer churn prediction model that improved retention by 15%.",
            "I approach problems systematically by breaking them down and researching solutions.",
            "I'm most comfortable with Python, pandas, scikit-learn, and TensorFlow.",
            "I quickly learned Apache Spark for a big data project by studying and practicing.",
            "The biggest challenge is staying current with AI advancements while ensuring ethics."
        ]
        self.answer_index = 0
    
    def create_mock_audio(self) -> str:
        """Create valid 0.2 second WAV audio"""
        sample_rate = 16000
        duration = 0.2
        samples = int(sample_rate * duration)
        
        import numpy as np
        t = np.linspace(0, duration, samples, False)
        frequency = 440
        audio_data = (np.sin(2 * np.pi * frequency * t) * 0.3 * 32767).astype(np.int16)
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        wav_buffer.seek(0)
        return base64.b64encode(wav_buffer.read()).decode('utf-8')
    
    def test_setup_interview(self):
        """Test setup-interview endpoint"""
        print("\n🔧 Testing /setup-interview endpoint...")
        
        payload = {
            "job_role": "Data Scientist",
            "job_description": "Senior data scientist role with ML expertise required",
            "resume_text": "John Smith - Data Scientist, 5 years experience"
        }
        
        try:
            response = requests.post(f"{INTERVIEW_URL}/setup-interview", json=payload)
            print(f"   Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                self.access_code = data["access_code"]
                print(f"✅ Setup successful!")
                print(f"   Access Code: {self.access_code}")
                print(f"   Message: {data['message']}")
                return True
            else:
                print(f"❌ Setup failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Setup error: {e}")
            return False
    
    def test_start_interview(self):
        """Test start-interview endpoint"""
        print(f"\n🚀 Testing /start-interview endpoint...")
        
        if not self.access_code:
            print("❌ No access code")
            return False
        
        payload = {"access_code": self.access_code}
        
        try:
            response = requests.post(f"{INTERVIEW_URL}/start-interview", json=payload)
            print(f"   Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Interview started!")
                print(f"   Question #{data['question_number']}: {data['question_text'][:80]}...")
                print(f"   Has audio: {len(data['question_audio']) > 0}")
                return True
            else:
                print(f"❌ Start failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Start error: {e}")
            return False
    
    def test_answer_question(self, answer_text: str):
        """Test answer-question endpoint"""
        print(f"\n🎤 Testing /answer-question endpoint...")
        print(f"   Answer: '{answer_text[:50]}...'")
        
        if not self.access_code:
            print("❌ No access code")
            return False
        
        payload = {
            "access_code": self.access_code,
            "audio_data": self.mock_audio_data
        }
        
        try:
            response = requests.post(f"{INTERVIEW_URL}/answer-question", json=payload)
            print(f"   Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("interview_completed", False):
                    print("✅ Answer processed - Interview completed!")
                    return {"success": True, "completed": True}
                else:
                    print(f"✅ Answer processed!")
                    print(f"   Next Question #{data['question_number']}: {data['question_text'][:80]}...")
                    return {"success": True, "completed": False}
            else:
                print(f"❌ Answer failed: {response.text}")
                return {"success": False, "completed": False}
                
        except Exception as e:
            print(f"❌ Answer error: {e}")
            return {"success": False, "completed": False}
    
    def run_quick_test(self):
        """Run a quick test of the new API"""
        print("🧪 Quick MongoDB API Test")
        print("=" * 50)
        
        # Test the server
        try:
            response = requests.get(f"{BASE_URL}/")
            data = response.json()
            print(f"✅ Server running: {data.get('message', 'Unknown')}")
            print(f"   Version: {data.get('version', 'Unknown')}")
            if 'features' in data:
                print(f"   Features: {', '.join(data['features'])}")
        except Exception as e:
            print(f"❌ Server not accessible: {e}")
            return False
        
        # Test setup
        if not self.test_setup_interview():
            return False
        
        # Test start
        if not self.test_start_interview():
            return False
        
        # Test a few answers
        for i in range(min(3, len(self.test_answers))):
            answer = self.test_answers[i]
            result = self.test_answer_question(answer)
            
            if not result["success"]:
                return False
            
            if result["completed"]:
                print(f"\n🎯 Interview completed after {i + 1} questions!")
                break
            
            time.sleep(1)
        
        print("\n" + "=" * 50)
        print("🎉 QUICK TEST PASSED!")
        print(f"   Access Code: {self.access_code}")
        print("   New MongoDB API is working!")
        return True

def main():
    """Main test runner"""
    tester = QuickMongoTest()
    success = tester.run_quick_test()
    
    if success:
        print("\n🎯 New MongoDB API is working correctly!")
        print("💡 You can now use the 5-digit access code system")
    else:
        print("\n💥 Test failed - check server logs")

if __name__ == "__main__":
    main()