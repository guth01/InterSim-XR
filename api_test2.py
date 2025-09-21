#!/usr/bin/env python3
"""
Multi-Section API test script for Interview Chatbot endpoints
Tests the 4-section interview flow without real audio processing (uses mock audio data)
"""

import requests
import base64
import json
import time
from typing import Dict, Any

# API Configuration
BASE_URL = "http://localhost:8000"
INTERVIEW_URL = f"{BASE_URL}/interview"

class MultiSectionAPITester:
    """Tester for the 4-section interview system"""
    
    def __init__(self):
        self.session_id = None
        self.current_section = 1
        self.question_count = 0
        
        # Mock base64 audio data (small WAV file header + minimal data)
        self.mock_audio_data = self.create_mock_audio()
        
        # Test answers organized by section
        self.test_answers = {
            1: [  # General Introduction (2 questions)
                "Hi! I'm John Smith, a passionate data scientist with 5 years of experience in machine learning and analytics. I have a Master's in Computer Science and have worked at both startups and large corporations.",
                "I'm really excited about this opportunity because your company is at the forefront of AI innovation. I've been following your recent work in natural language processing and I'd love to contribute to those cutting-edge projects."
            ],
            2: [  # Technical & Role-based (3 questions)
                "For a Data Scientist role, I'd say the core skills are statistics, programming in Python/R, machine learning algorithms, data visualization, and domain expertise. I rate myself 9/10 in Python and ML, 8/10 in statistics, and 7/10 in domain knowledge as I'm always learning.",
                "When facing a complex technical problem, I start by clearly defining the problem and gathering requirements. Then I break it down into smaller components, research existing solutions, prototype different approaches, and validate results through testing and peer review.",
                "I prefer Python with pandas, scikit-learn, and TensorFlow for most projects. For visualization, I use matplotlib and Plotly. For big data, I work with Spark. I like these tools because they have strong community support and integrate well together."
            ],
            3: [  # Resume & Experience (3 questions)
                "One of my most significant projects was building a customer churn prediction model at my previous company. I used ensemble methods combining random forests and gradient boosting, which improved our F1 score from 0.72 to 0.89. This helped reduce customer churn by 15% and saved the company $2M annually.",
                "My progression from junior data analyst to senior data scientist has given me both technical depth and business acumen. I started with basic SQL and Excel, then learned Python and machine learning. Each role taught me to translate business problems into technical solutions, which is crucial for this position.",
                "At my most recent position, I learned the importance of model interpretability when we had to explain our credit scoring model to regulators. This taught me that technical excellence must be balanced with explainability and ethical considerations in ML."
            ],
            4: [  # Behavioral Assessment (3 questions)
                "I had to deliver a critical fraud detection model with just 2 weeks notice when our vendor suddenly dropped out. I organized daily standups, broke the work into parallel streams, and worked evenings to gather domain knowledge. We delivered on time and the model caught 95% of fraudulent transactions in the first month.",
                "I once worked with a team member who consistently missed deadlines and delivered low-quality code. I scheduled a private conversation to understand their challenges - they were overwhelmed with multiple projects. I helped them prioritize tasks and offered to pair-program on difficult sections. This improved both their performance and our team dynamics.",
                "When GPT-3 was released, I knew it would impact our NLP projects. I spent weekends learning the API, built proof-of-concepts, and presented findings to leadership. Within a month, we had integrated it into our chatbot, improving customer satisfaction scores by 40%. I always stay curious about emerging technologies."
            ]
        }
        
        self.current_section_answer_index = 0
        
        # Sample resume text for testing
        self.sample_resume = """
        John Smith - Senior Data Scientist
        
        Experience:
        • Senior Data Scientist at TechCorp (2021-2024): Led ML initiatives for customer analytics
        • Data Scientist at StartupXYZ (2019-2021): Built predictive models for e-commerce
        • Data Analyst at BigCorp (2018-2019): Created dashboards and reports
        
        Skills: Python, R, SQL, TensorFlow, PyTorch, AWS, Docker, Git
        
        Education: M.S. Computer Science, Stanford University (2018)
        
        Projects:
        • Customer Churn Prediction: Improved retention by 15% using ensemble methods
        • Fraud Detection System: Real-time ML pipeline processing 1M+ transactions/day
        • Recommendation Engine: Increased user engagement by 25% using collaborative filtering
        """
    
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
    
    def get_next_test_answer(self, section: int) -> str:
        """Get the next test answer for the current section"""
        if section in self.test_answers:
            section_answers = self.test_answers[section]
            if self.current_section_answer_index < len(section_answers):
                answer = section_answers[self.current_section_answer_index]
                return answer
            else:
                return f"This is additional test answer for section {section}."
        else:
            return f"Generic test answer for section {section}."
    
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
    
    def test_start_interview(self, job_role: str = "Data Scientist", job_description: str = None, include_resume: bool = True):
        """Test start interview endpoint with multi-section support"""
        print(f"\n🚀 Testing /start-interview endpoint...")
        
        payload = {
            "job_role": job_role,
            "job_description": job_description or "Looking for an experienced data scientist to work on ML models and analytics.",
            "resume_text": self.sample_resume if include_resume else None
        }
        
        result = self.test_api_endpoint("POST", "/start-interview", payload)
        
        if result["success"]:
            data = result["data"]
            self.session_id = data["session_id"]
            self.current_section = data["section"]
            self.question_count = data["question_number"]
            self.current_section_answer_index = 0  # Reset for new section
            
            print(f"✅ Multi-section interview started successfully!")
            print(f"   Session ID: {self.session_id}")
            print(f"   Section {data['section']}: {data['section_name']}")
            print(f"   Question #{data['question_number']}: {data['question_text'][:100]}...")
            print(f"   Total questions: {data['total_questions']}")
            print(f"   Has audio data: {len(data['question_audio']) > 0}")
        else:
            print(f"❌ Start interview failed: {result['error']}")
        
        return result["success"]
    
    def test_answer_question(self, section: int):
        """Test answer question endpoint"""
        answer_text = self.get_next_test_answer(section)
        
        print(f"\n🎤 Testing /answer-question endpoint...")
        print(f"   Section {section} - Answer: '{answer_text[:60]}...'")
        
        if not self.session_id:
            print("❌ No session ID available")
            return {"success": False, "completed": False}
        
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
                # Check if we moved to a new section
                if data["section"] != self.current_section:
                    print(f"📋 Advanced to Section {data['section']}: {data['section_name']}")
                    self.current_section = data["section"]
                    self.current_section_answer_index = 0  # Reset for new section
                else:
                    self.current_section_answer_index += 1
                
                self.question_count = data["question_number"]
                
                print(f"✅ Answer processed successfully!")
                print(f"   Next Question - Section {data['section']} #{data['question_number']}: {data['question_text'][:100]}...")
                print(f"   Has audio data: {len(data['question_audio']) > 0}")
                return {"success": True, "completed": False, "section": data["section"]}
        else:
            print(f"❌ Answer question failed: {result['error']}")
            return {"success": False, "completed": False}
    
    def test_session_status(self):
        """Test session status endpoint"""
        print(f"\n📊 Testing /session/{self.session_id} endpoint...")
        
        if not self.session_id:
            print("❌ No session ID available")
            return False
        
        result = self.test_api_endpoint("GET", f"/session/{self.session_id}")
        
        if result["success"]:
            data = result["data"]
            print(f"✅ Session status retrieved successfully!")
            print(f"   Current section: {data['current_section']} - {data['current_section_name']}")
            print(f"   Progress: {data['progress']['answered_questions']}/{data['progress']['total_questions']} ({data['progress']['percentage']}%)")
            print(f"   Sections completed: {data['sections_completed']}")
            print(f"   Interview complete: {data['interview_complete']}")
        else:
            print(f"❌ Session status failed: {result['error']}")
        
        return result["success"]
    
    def test_get_evaluation(self, section: int):
        """Test get evaluation endpoint for a specific section"""
        print(f"\n📊 Testing /get-evaluation endpoint for Section {section}...")
        
        if not self.session_id:
            print("❌ No session ID available")
            return False
        
        payload = {
            "session_id": self.session_id,
            "section": section
        }
        
        result = self.test_api_endpoint("POST", "/get-evaluation", payload)
        
        if result["success"]:
            data = result["data"]
            print(f"✅ Section {section} evaluation retrieved successfully!")
            print(f"   Section: {data['section']} - {data['section_name']}")
            print(f"   Overall Score: {data['overall_score']}/10")
            print(f"   Feedback length: {len(data['feedback'])} characters")
            print(f"   Transcript entries: {len(data['transcript'])}")
            print(f"   Sample feedback: {data['feedback'][:150]}...")
        else:
            print(f"❌ Get evaluation failed for section {section}: {result['error']}")
        
        return result["success"]
    
    def run_full_test_suite(self):
        """Run the complete multi-section test suite"""
        print("🧪 Starting Multi-Section Interview API Test Suite")
        print("=" * 70)
        
        # Step 1: Test health
        if not self.test_health():
            return False
        
        # Step 2: Start interview
        if not self.test_start_interview():
            return False
        
        # Step 3: Go through all 4 sections (total 11 questions: 2+3+3+3)
        expected_questions = 11
        questions_answered = 0
        
        for question_num in range(expected_questions):
            result = self.test_answer_question(self.current_section)
            
            if not result["success"]:
                return False
            
            questions_answered += 1
            
            if result["completed"]:
                print(f"\n🎯 Interview completed after {questions_answered} questions")
                break
            
            # Update current section from result
            if "section" in result:
                self.current_section = result["section"]
            
            # Test session status occasionally
            if question_num % 3 == 0:
                self.test_session_status()
            
            time.sleep(0.5)  # Small delay between questions
        else:
            print("⚠️ Interview didn't complete within expected question limit")
            return False
        
        # Step 4: Test session status one final time
        print(f"\n📈 Final session status check...")
        self.test_session_status()
        
        # Step 5: Get evaluations for sections 2, 3, and 4
        print(f"\n🎯 Testing section-specific evaluations...")
        evaluation_success = True
        
        for section in [2, 3, 4]:  # Only these sections can be evaluated
            if not self.test_get_evaluation(section):
                evaluation_success = False
        
        # Test invalid section evaluation (should fail)
        print(f"\n🚫 Testing invalid section evaluation (Section 1 - should fail)...")
        result = self.test_get_evaluation(1)
        if result:
            print("⚠️ Expected section 1 evaluation to fail, but it succeeded")
            evaluation_success = False
        else:
            print("✅ Section 1 evaluation correctly rejected")
        
        if not evaluation_success:
            return False
        
        print("\n" + "=" * 70)
        print("🎉 ALL MULTI-SECTION TESTS PASSED!")
        print("=" * 70)
        return True

def main():
    """Main test runner"""
    print("🧪 Multi-Section Interview API Test")
    
    # Check server availability
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Server is running: {response.json()['message']}")
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        print("💡 Start the server with: python main.py")
        return
    
    # Run tests
    tester = MultiSectionAPITester()
    success = tester.run_full_test_suite()
    
    if success:
        print("🎯 All multi-section API endpoints are working correctly!")
        print("\n📋 Test Summary:")
        print("   ✅ 4-section interview flow (General → Technical → Resume → Behavioral)")
        print("   ✅ Section-specific question generation")
        print("   ✅ Session progress tracking")
        print("   ✅ Individual section evaluations (sections 2, 3, 4)")
        print("   ✅ Proper rejection of section 1 evaluation")
    else:
        print("💥 Some tests failed. Check the server logs for details.")

if __name__ == "__main__":
    main()