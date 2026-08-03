"""
FastAPI web application for SR-MARE with chatbot UI.
"""

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import uvicorn
import asyncio
import concurrent.futures
import json

from sr_mare.core.orchestrator import ResearchOrchestrator

app = FastAPI(title="SR-MARE", description="Self-Reflective Multi-Agent Research Engine")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Global orchestrator instance
orchestrator = None


class ResearchRequest(BaseModel):
    question: str


class DocumentsRequest(BaseModel):
    documents: List[str]


import os

def initialize_orchestrator():
    """Initialize the research orchestrator."""
    global orchestrator
    
    if orchestrator is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        orchestrator = ResearchOrchestrator(
            groq_api_key=api_key,
            groq_model="llama-3.1-8b-instant",
            max_iterations=3,
            confidence_threshold=0.75
        )
        
        # Load default documents
        doc_path = Path("sr_mare/data/documents.txt")
        if doc_path.exists():
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            documents = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
            orchestrator.load_documents(documents)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main chatbot interface."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/health")
async def health_check():
    """Check if the system is ready."""
    try:
        initialize_orchestrator()
        connections_ok = orchestrator.test_connections()
        
        stats = orchestrator.vector_store.get_stats()
        
        return {
            'status': 'ready' if connections_ok else 'error',
            'connections': connections_ok,
            'documents_loaded': stats['total_documents'],
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/research")
async def research(request: ResearchRequest):
    """Process a research question."""
    try:
        question = request.question.strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")
        
        initialize_orchestrator()
        
        # Run research
        result = orchestrator.research(question, top_k=5)
        
        # Format response for UI
        response = {
            'success': True,
            'question': result['question'],
            'answer': result['final_answer'],
            'confidence': result['confidence_score'],
            'metrics': {
                'critic_score': result['confidence_metrics']['critic_quality_score'],
                'self_consistency': result['confidence_metrics']['self_consistency_score'],
                'evidence_diversity': result['confidence_metrics']['evidence_diversity_score'],
                'retrieval_quality': result['confidence_metrics']['retrieval_quality']
            },
            'iterations': result['iterations'],
            'sources': [
                {
                    'text': src['text'],
                    'similarity': src['similarity']
                }
                for src in result['retrieved_sources'][:3]
            ],
            'critique': {
                'strengths': result['critic_feedback'].get('strengths', []),
                'weaknesses': result['critic_feedback'].get('weaknesses', []),
                'hallucination_risk': result['critic_feedback'].get('hallucination_risk', 'unknown')
            },
            'duration': result['duration_seconds'],
            'timestamp': result['timestamp']
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/research")
async def websocket_research(websocket: WebSocket):
    await websocket.accept()
    initialize_orchestrator()
    
    # Queue for user responses (MCQ)
    user_response_queue = asyncio.Queue()
    
    # Task to read from websocket
    async def receive_messages():
        try:
            while True:
                data = await websocket.receive_text()
                await user_response_queue.put(json.loads(data))
        except WebSocketDisconnect:
            pass
            
    receive_task = asyncio.create_task(receive_messages())
    
    try:
        # Wait for initial query
        initial_data = await user_response_queue.get()
        question = initial_data.get("question", "").strip()
        
        if not question:
            await websocket.send_json({"type": "error", "message": "No question provided"})
            return
            
        loop = asyncio.get_running_loop()
        
        def sync_event_callback(event_data):
            # Send status update safely from background thread
            asyncio.run_coroutine_threadsafe(websocket.send_json(event_data), loop)
            
        def sync_mcq_callback(mcq_data):
            # Request MCQ response from user
            asyncio.run_coroutine_threadsafe(websocket.send_json(mcq_data), loop)
            # Block and wait for answer (must use a thread-safe way to get from queue)
            # We use asyncio.run_coroutine_threadsafe to await the queue.get()
            future = asyncio.run_coroutine_threadsafe(user_response_queue.get(), loop)
            response = future.result()  # Blocks the thread until user answers
            return response.get("answer")
            
        # Run orchestrator in thread pool to not block asyncio event loop
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(
                pool,
                lambda: orchestrator.research(
                    question, 
                    top_k=5, 
                    event_callback=sync_event_callback, 
                    mcq_callback=sync_mcq_callback
                )
            )
            
        # Format response for UI
        response = {
            'type': 'result',
            'success': True,
            'question': result['question'],
            'answer': result['final_answer'],
            'confidence': result['confidence_score'],
            'metrics': {
                'critic_score': result['confidence_metrics']['critic_quality_score'],
                'self_consistency': result['confidence_metrics']['self_consistency_score'],
                'evidence_diversity': result['confidence_metrics']['evidence_diversity_score'],
                'retrieval_quality': result['confidence_metrics']['retrieval_quality']
            },
            'iterations': result['iterations'],
            'sources': [
                {
                    'text': src['text'],
                    'similarity': src['similarity']
                }
                for src in result['retrieved_sources'][:3]
            ],
            'critique': {
                'strengths': result['critic_feedback'].get('strengths', []),
                'weaknesses': result['critic_feedback'].get('weaknesses', []),
                'hallucination_risk': result['critic_feedback'].get('hallucination_risk', 'unknown')
            },
            'duration': result['duration_seconds'],
            'timestamp': result['timestamp']
        }
        
        await websocket.send_json(response)
        
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        receive_task.cancel()


@app.post("/api/upload_documents")
async def upload_documents(request: DocumentsRequest):
    """Upload custom documents."""
    try:
        documents = request.documents
        
        if not documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        initialize_orchestrator()
        orchestrator.load_documents(documents)
        
        stats = orchestrator.vector_store.get_stats()
        
        return {
            'success': True,
            'message': f'Loaded {len(documents)} documents',
            'total_documents': stats['total_documents']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    try:
        initialize_orchestrator()
        
        vector_stats = orchestrator.vector_store.get_stats()
        summary_stats = orchestrator.get_summary_statistics()
        
        return {
            'vector_store': vector_stats,
            'experiments': summary_stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    print("🚀 Starting SR-MARE Web Interface...")
    print("📱 Open your browser at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
