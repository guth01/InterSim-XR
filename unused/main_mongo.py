# main_mongo.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the MongoDB-based interview router
from interview_router_mongo import router as interview_router

# Create FastAPI app
app = FastAPI(
    title="Interview Chatbot API with MongoDB",
    description="Conversational AI interviewer with audio support and MongoDB Atlas integration",
    version="2.0.0"
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
    return {
        "message": "Interview Chatbot API with MongoDB is running", 
        "version": "2.0.0",
        "features": [
            "5-digit access codes",
            "MongoDB Atlas integration", 
            "Voice metrics analysis",
            "Confidence scoring",
            "Report generation"
        ]
    }

# Health check
@app.get("/health")
def health():
    return {"status": "healthy", "database": "mongodb"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_mongo:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )