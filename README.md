# SR-MARE: Self-Reflective Multi-Agent Research Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A fully local, Ollama-powered multi-agent research system that breaks down complex questions, retrieves evidence, generates hypotheses, critically evaluates its own answers, and iteratively refines responses with confidence scoring.

## 🎯 Features

- **Multi-Agent Architecture**: Specialized agents for planning, analysis, criticism, and refinement
- **Vector-Based Retrieval**: FAISS-powered semantic search with Ollama embeddings
- **Self-Reflection**: Critic agent evaluates answer quality and identifies weaknesses
- **Uncertainty Quantification**: Multiple metrics including self-consistency, evidence diversity, and token entropy
- **Iterative Refinement**: Automatically improves answers until confidence threshold is met
- **Fully Local**: No external APIs required - runs entirely on your machine with Ollama
- **Research-Grade Metrics**: Comprehensive evaluation including calibration, retrieval quality, and iteration improvement

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Research Question                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ Planner Agent  │ (mistral)
              │ Task Breakdown │
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │    Retrieval   │ (nomic-embed-text + FAISS)
              │  Vector Search │
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │ Analyst Agent  │ (mistral)
              │ Generate Answer│
              │  + Hypotheses  │
              └────────┬───────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │                              │
        │   Iterative Refinement Loop  │
        │   (until confidence >= 0.75  │
        │    or max_iterations = 3)    │
        │                              │
        │  ┌────────────────┐          │
        │  │  Critic Agent  │ (llama3.2) │
        │  │   Evaluation   │          │
        │  └───────┬────────┘          │
        │          │                   │
        │          ▼                   │
        │  ┌────────────────┐          │
        │  │  Uncertainty   │          │
        │  │   Estimator    │          │
        │  └───────┬────────┘          │
        │          │                   │
        │          ▼                   │
        │  ┌────────────────┐          │
        │  │ Refiner Agent  │ (llama3.2) │
        │  │ Improve Answer │          │
        │  └───────┬────────┘          │
        │          │                   │
        └──────────┼───────────────────┘
                   │
                   ▼
          ┌────────────────┐
          │ Final Answer   │
          │ + Confidence   │
          │ + Full Report  │
          └────────────────┘
```

## 📁 Project Structure

```
sr_mare/
│
├── agents/
│   ├── __init__.py
│   ├── planner.py         # Task decomposition (mistral)
│   ├── analyst.py         # Answer generation + hypotheses (mistral)
│   ├── critic.py          # Quality evaluation (llama3.2)
│   └── refiner.py         # Answer improvement (llama3.2)
│
├── retrieval/
│   ├── __init__.py
│   ├── embedder.py        # Ollama embedding generation
│   └── vector_store.py    # FAISS vector search
│
├── evaluation/
│   ├── __init__.py
│   ├── uncertainty.py     # Confidence scoring
│   └── metrics.py         # Research-grade metrics
│
├── core/
│   ├── __init__.py
│   └── orchestrator.py    # Pipeline coordinator
│
├── data/
│   ├── __init__.py
│   └── documents.txt      # Sample knowledge base
│
├── main.py                # CLI interface
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Installation

### Prerequisites

1. **Python 3.8+**
2. **Ollama** installed and running

### Step 1: Install Ollama

Download and install Ollama from [https://ollama.ai](https://ollama.ai)

### Step 2: Pull Required Models

```bash
# Pull the models used by SR-MARE
ollama pull mistral
ollama pull llama3.2
ollama pull nomic-embed-text
```

Verify models are installed:
```bash
ollama list
```

### Step 3: Install Python Dependencies

```bash
# Clone or navigate to the project directory
cd "d:\Research agent"

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Test Ollama connection
python main.py --test-connection
```

You should see:
```
✓ Embedder connection successful
✓ Planner connection successful
✓ Critic connection successful
✓ All connections successful
```

## 🎮 Usage

### Basic Usage

```bash
python main.py --question "What are the main causes of climate change?"
```

### Interactive Mode

```bash
python main.py
# You'll be prompted to enter your question
```

### Advanced Options

```bash
# Custom documents file
python main.py --question "Your question" --documents path/to/docs.txt

# Adjust confidence threshold
python main.py --question "Your question" --threshold 0.8

# Change max iterations
python main.py --question "Your question" --max-iterations 5

# Retrieve more documents
python main.py --question "Your question" --top-k 10

# Custom output file
python main.py --question "Your question" --output my_results.json

# Custom Ollama URL
python main.py --question "Your question" --ollama-url http://localhost:11434
```

### Full Command-Line Options

```
usage: main.py [-h] [--question QUESTION] [--documents DOCUMENTS]
               [--output OUTPUT] [--threshold THRESHOLD]
               [--max-iterations MAX_ITERATIONS] [--top-k TOP_K]
               [--ollama-url OLLAMA_URL] [--test-connection]

Options:
  -h, --help            Show help message
  --question, -q        Research question to answer
  --documents, -d       Path to documents file (default: sr_mare/data/documents.txt)
  --output, -o          Output file path (default: results.json)
  --threshold, -t       Confidence threshold (default: 0.75)
  --max-iterations, -m  Maximum refinement iterations (default: 3)
  --top-k, -k          Number of documents to retrieve (default: 5)
  --ollama-url         Ollama API URL (default: http://localhost:11434)
  --test-connection    Test Ollama connection and exit
```

## 🌐 Web Interface (Chatbot UI)

SR-MARE includes a beautiful, modern web interface with a chatbot-style UI for interactive research sessions.

### Start the Web Interface

**Option 1: Quick Start**
```bash
python start_web.py
```

**Option 2: Direct Launch**
```bash
python app.py
```

**Option 3: With Auto-Reload (Development)**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open your browser at: **http://localhost:8000**

### Web UI Features

- 🎨 **Modern Dark Theme** - Beautiful gradient design with smooth animations
- 💬 **Chat Interface** - Natural conversation flow with message bubbles
- 📊 **Visual Metrics** - Real-time confidence scores and quality assessments
- 🔍 **Source Attribution** - View retrieved documents with similarity scores
- 📈 **Live Statistics** - Track system status and performance metrics
- ⚡ **Fast API** - Built on FastAPI for high performance
- 📱 **Responsive Design** - Works on desktop, tablet, and mobile

### Web API Endpoints

The web interface exposes these REST API endpoints:

- `GET /` - Serve the web UI
- `GET /api/health` - Check system status
- `POST /api/research` - Submit research question
- `POST /api/upload_documents` - Upload custom documents
- `GET /api/stats` - Get system statistics

### Example API Usage

```python
import requests

# Submit a research question
response = requests.post('http://localhost:8000/api/research', json={
    'question': 'What are the main causes of climate change?'
})

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 📊 Output Format

The system generates a comprehensive JSON report with the following structure:

```json
{
  "question": "Your research question",
  "task_breakdown": {
    "key_concepts": ["concept1", "concept2"],
    "subtasks": ["subtask1", "subtask2"],
    "retrieval_strategy": "Search strategy description",
    "challenges": "Potential difficulties"
  },
  "final_answer": "The refined, high-confidence answer",
  "confidence_score": 0.82,
  "confidence_metrics": {
    "final_confidence": 0.82,
    "critic_quality_score": 0.85,
    "self_consistency_score": 0.78,
    "evidence_diversity_score": 0.65,
    "token_entropy": 0.58,
    "retrieval_quality": 0.73
  },
  "critic_feedback": {
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "hallucination_risk": "low",
    "logical_gaps": "Description of gaps",
    "quality_score": 0.85,
    "improvement_suggestions": "Specific suggestions"
  },
  "alternative_hypotheses": ["hypothesis1", "hypothesis2", "hypothesis3"],
  "iterations": 2,
  "iteration_metrics": {
    "total_improvement": 0.15,
    "avg_improvement_per_iteration": 0.075,
    "converged": true
  },
  "retrieved_sources": [
    {
      "text": "Relevant document excerpt...",
      "similarity": 0.87,
      "metadata": {"doc_id": 0}
    }
  ],
  "duration_seconds": 45.3,
  "timestamp": "2026-02-22T10:30:45.123456"
}
```

## 🧪 Uncertainty Scoring

SR-MARE implements multiple uncertainty estimation techniques:

### 1. Self-Consistency Score
- Generates multiple independent hypotheses (temperature=0.8)
- Computes token-level Jaccard similarity between hypotheses
- Higher score = more agreement = higher confidence

### 2. Evidence Diversity Score
- Analyzes retrieved documents for vocabulary overlap
- Measures information diversity in source material
- Moderate diversity is ideal for confidence

### 3. Token Entropy
- Computes Shannon entropy of token distribution
- Normalized to [0, 1] range
- Moderate entropy indicates balanced, non-repetitive text

### 4. Critic Quality Score
- LLM-based evaluation of answer quality
- Assesses factual accuracy, logical coherence, completeness
- Identifies hallucination risks and logical gaps

### 5. Retrieval Quality
- Average similarity score of top-k retrieved documents
- Higher scores = better evidence support

### Final Confidence Formula

```
confidence = 0.35 × critic_score +
             0.25 × self_consistency +
             0.20 × retrieval_quality +
             0.10 × entropy_factor +
             0.10 × evidence_diversity
```

Where `entropy_factor = 1 - |entropy - 0.6|` (penalizes extreme values)

## 🔄 Iterative Refinement

The system automatically refines answers through multiple iterations:

1. **Initial Generation**: Analyst creates answer + 3 hypotheses
2. **Evaluation**: Critic assesses quality and identifies issues
3. **Confidence Check**: If confidence < threshold, proceed to refinement
4. **Refinement**: Refiner improves answer based on critic feedback
5. **Repeat**: Continue until confidence ≥ 0.75 or max_iterations (default: 3)

### Stopping Criteria

- **Confidence Threshold**: Stop when confidence ≥ 0.75 (configurable)
- **Max Iterations**: Stop after 3 iterations (configurable)
- **Convergence**: System detects when improvements plateau

## 📈 Research-Grade Metrics

### Iteration Improvement
- Total improvement across iterations
- Average improvement per iteration
- Convergence detection

### Retrieval Metrics
- Number of documents retrieved
- Average similarity scores
- Relevance ratio (above threshold)

### Calibration Metrics
- Alignment between predicted confidence and actual quality
- Overconfidence/underconfidence detection
- Calibration error quantification

### Answer Stability
- Variance across multiple runs
- Reproducibility assessment

## 🗂️ Adding Your Own Documents

### Format 1: Paragraph-separated (recommended)

Create a text file with documents separated by double newlines:

```
Document 1 text here.
This can span multiple lines.

Document 2 text here.
Another paragraph of information.

Document 3 text here.
And so on.
```

### Format 2: Line-separated

One document per line:

```
Document 1 on a single line
Document 2 on another line
Document 3 here
```

### Usage

```bash
python main.py --documents path/to/your/docs.txt --question "Your question"
```

The system will:
1. Load and split documents
2. Generate embeddings using `nomic-embed-text`
3. Index in FAISS for fast similarity search
4. Retrieve top-k most relevant documents for your question

## 🧑‍💻 Programmatic Usage

```python
from sr_mare.core.orchestrator import ResearchOrchestrator

# Initialize
orchestrator = ResearchOrchestrator(
    base_url="http://localhost:11434",
    max_iterations=3,
    confidence_threshold=0.75
)

# Load documents
documents = [
    "Document 1 text...",
    "Document 2 text...",
    "Document 3 text..."
]
orchestrator.load_documents(documents)

# Run research
result = orchestrator.research(
    question="What are the main causes of climate change?",
    top_k=5
)

# Access results
print(f"Answer: {result['final_answer']}")
print(f"Confidence: {result['confidence_score']:.3f}")
print(f"Iterations: {result['iterations']}")

# Save results
orchestrator.save_result(result, "output.json")
```

## 🔧 Configuration

### Model Selection

Edit the orchestrator initialization to use different models:

```python
from sr_mare.agents.planner import PlannerAgent
from sr_mare.agents.analyst import AnalystAgent

# Use different models
planner = PlannerAgent(model="llama3.2")  # Instead of mistral
analyst = AnalystAgent(model="mixtral")  # Instead of mistral
```

### Embedding Dimension

If using a different embedding model, adjust the dimension:

```python
from sr_mare.retrieval.vector_store import FAISSVectorStore

vector_store = FAISSVectorStore(dimension=384)  # For smaller models
```

### Confidence Weights

Adjust uncertainty scoring weights in `sr_mare/evaluation/uncertainty.py`:

```python
weights = {
    "critic_score": 0.40,        # Increase critic importance
    "self_consistency": 0.30,
    "retrieval_quality": 0.15,
    "entropy_factor": 0.10,
    "evidence_diversity": 0.05
}
```

## 🧪 Research & Experiments

### Running Experiments

```python
from sr_mare.core.orchestrator import ResearchOrchestrator

orchestrator = ResearchOrchestrator()
orchestrator.load_documents(documents)

questions = [
    "What causes climate change?",
    "How does machine learning work?",
    "What is quantum computing?"
]

for question in questions:
    result = orchestrator.research(question)
    orchestrator.save_result(result, f"experiment_{i}.json")

# Get aggregate statistics
stats = orchestrator.get_summary_statistics()
print(stats)
```

### Analyzing Results

```python
import json

# Load multiple results
results = []
for i in range(10):
    with open(f"experiment_{i}.json") as f:
        results.append(json.load(f))

# Analyze confidence distribution
confidences = [r["confidence_score"] for r in results]
print(f"Mean confidence: {sum(confidences)/len(confidences):.3f}")

# Analyze iteration patterns
iterations = [r["iterations"] for r in results]
print(f"Mean iterations: {sum(iterations)/len(iterations):.1f}")
```

## 🐛 Troubleshooting

### "Connection failed" errors

1. Verify Ollama is running:
   ```bash
   ollama serve
   ```

2. Check if models are installed:
   ```bash
   ollama list
   ```

3. Test individual model:
   ```bash
   ollama run mistral
   ```

### "No documents loaded" warning

- Verify the documents file path exists
- Check file has content and proper formatting
- Use `--documents` flag to specify correct path

### Slow performance

- **Reduce iterations**: `--max-iterations 2`
- **Reduce retrieval**: `--top-k 3`
- **Use smaller models**: Edit agent initialization
- **Use GPU**: Install `faiss-gpu` instead of `faiss-cpu`

### Memory issues with large document sets

- Split documents into smaller batches
- Use approximate FAISS indices (edit `vector_store.py`)
- Reduce embedding dimension if using custom model

## 📚 Dependencies

- **requests**: Ollama API communication
- **numpy**: Numerical computations
- **faiss-cpu**: Vector similarity search
- **Python 3.8+**: Core runtime

Optional:
- **faiss-gpu**: GPU-accelerated vector search
- **pytest**: Testing framework

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional agent types (fact-checker, summarizer)
- Advanced retrieval strategies (reranking, query expansion)
- More uncertainty metrics (epistemic vs aleatoric)
- Experiment tracking integration (mlflow, wandb)
- Web interface
- Support for other LLM backends

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **Ollama**: Local LLM inference engine
- **FAISS**: Efficient similarity search library
- **Research inspiration**: Self-consistency prompting, Constitutional AI, uncertainty estimation in LLMs

## 📞 Support

For issues, questions, or suggestions:
1. Check the troubleshooting section
2. Review closed issues on GitHub
3. Open a new issue with:
   - Python version
   - Ollama version
   - Model versions
   - Error messages
   - Minimal reproduction example

## 🗺️ Roadmap

- [ ] Multi-document synthesis
- [ ] Citation tracking and source attribution
- [ ] Adversarial evaluation agent
- [ ] Knowledge graph construction
- [ ] Experiment dashboard
- [ ] Automated hyperparameter tuning
- [ ] Support for multimodal inputs
- [ ] Integration with external knowledge bases

---

**Built with ❤️ for research-grade local AI systems**
