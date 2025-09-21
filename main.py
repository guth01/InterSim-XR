# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your interview router
from interview_router2 import router as interview_router  # Assuming the previous code is in interview_router.py

# Create FastAPI app
app = FastAPI(
    title="Interview Chatbot API",
    description="Conversational AI interviewer with audio support",
    version="1.0.0"
)

# Add CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(interview_router)

# Root endpoint
@app.get("/")
def root():
    return {"message": "Interview Chatbot API is running", "version": "1.0.0"}

# Health check
@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )