# SR-MARE: Self-Reflective Multi-Agent Research Engine

SR-MARE is a next-generation autonomous research engine capable of solving complex, multi-hop reasoning tasks with high accuracy, adaptability, transparency, and reliability. It goes beyond standard RAG (Retrieval-Augmented Generation) frameworks by employing an entropy-based Meta-Router, Fractal Retrieval loops, and mathematically grounded Bayesian Evidential Reasoning (Dempster-Shafer theory) to aggressively reduce LLM hallucinations.

## 🌟 Key Features

*   **Meta-Router (System 1 vs System 2 Thinking):** Evaluates the entropy (complexity) of your query. Simple questions bypass the swarm for instant RAG retrieval (Fast Lane), while complex questions spin up the full Deep Research swarm.
*   **Multi-Agent Swarm:** Specialized AI personas (Planner, Analyst, Forager, Critic, Refiner) work collaboratively to break down, research, debate, and refine answers.
*   **Fractal Retrieval:** Dynamic `exclude_indices` searching within the Vector Database (FAISS). The Forager agent autonomously hunts for missing context without re-reading the same documents.
*   **Epistemic Uncertainty Calibration:** Fuses retrieval quality, token-level semantic consistency (Jaccard), and qualitative Critic evaluations using Dempster-Shafer theory. High conflict actively tanks the confidence score, mathematically preventing silent hallucinations.
*   **Episodic Failure Memory:** The swarm logs failed hypotheses to a persistent Memory Bus, ensuring it never makes the same logical mistake twice.
*   **Real-time Streaming UI:** A sleek, glassmorphism UI powered by WebSockets streams the internal thoughts and status of the agent swarm directly to the browser.
*   **Manual Knowledge Base Uploads:** Drag-and-drop `.txt` or `.md` files directly in the UI to seamlessly update the FAISS vector database on the fly.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Arun-Prasad2004/Research-Agent.git
cd Research-Agent
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Ensure you have libraries like `fastapi`, `uvicorn`, `sentence-transformers`, `faiss-cpu`, and `groq` installed)*

### 4. Configure Your Groq API Key
This project utilizes the blazing-fast Llama 3 models via the Groq API. You must set your API key as an environment variable before starting the server.

**Get an API Key:** Sign up at [console.groq.com](https://console.groq.com) to generate a free API key.

**Set the Environment Variable:**
*   **Windows (Command Prompt):**
    ```cmd
    set GROQ_API_KEY=your_api_key_here
    ```
*   **Windows (PowerShell):**
    ```powershell
    $env:GROQ_API_KEY="your_api_key_here"
    ```
*   **macOS / Linux:**
    ```bash
    export GROQ_API_KEY="your_api_key_here"
    ```

### 5. Run the Server
Start the backend FastAPI server using Uvicorn:
```bash
python -m uvicorn app:app --reload
```

### 6. Access the Interface
Open your web browser and navigate to:
```
http://localhost:8000
```

---

## 🛠️ Usage

1.  **Chat Interface:** Ask complex questions. Watch the real-time status box to see exactly what the Meta-Router, Forager, and Critic agents are doing. Review the final confidence score and the evaluation metrics grid.
2.  **Knowledge Base:** Click the 📚 Knowledge Base tab in the sidebar. Drag and drop research papers (`.txt` or `.md`) or paste raw text to instantly index them into the Vector Database for the swarm to use in future queries.

---

## 🧠 Architecture Overview

*   **Backend:** `Python`, `FastAPI`, `WebSockets`, `concurrent.futures.ThreadPoolExecutor`
*   **AI Models:** `llama-3.1-8b-instant` (via Groq API)
*   **Vector Database:** `FAISS` (Facebook AI Similarity Search)
*   **Embeddings:** `SentenceTransformers` (`all-MiniLM-L6-v2`)
*   **Frontend:** Vanilla JS, HTML5, CSS3, `marked.js` (for Markdown rendering)
