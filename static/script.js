// Global state
let isProcessing = false;
let totalQueries = 0;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    checkSystemHealth();
    loadStats();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    const chatForm = document.getElementById('chatForm');
    const clearChatBtn = document.getElementById('clearChat');
    const exampleBtns = document.querySelectorAll('.example-btn');
    const questionInput = document.getElementById('questionInput');
    
    chatForm.addEventListener('submit', handleSubmit);
    clearChatBtn.addEventListener('click', clearChat);
    
    exampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const question = btn.dataset.question;
            questionInput.value = question;
            questionInput.focus();
        });
    });
    
    // Auto-resize textarea
    questionInput.addEventListener('input', () => {
        questionInput.style.height = 'auto';
        questionInput.style.height = questionInput.scrollHeight + 'px';
    });
}

// Check system health
async function checkSystemHealth() {
    const statusIndicator = document.querySelector('.status-indicator');
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');
    
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (data.status === 'ready') {
            statusDot.classList.add('ready');
            statusText.textContent = '✓ System Ready';
            document.getElementById('docCount').textContent = data.documents_loaded;
        } else {
            statusDot.classList.add('error');
            statusText.textContent = '✗ System Error';
        }
    } catch (error) {
        statusDot.classList.add('error');
        statusText.textContent = '✗ Connection Failed';
        console.error('Health check failed:', error);
    }
}

// Load statistics
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        if (data.experiments && data.experiments.total_experiments > 0) {
            document.getElementById('queryCount').textContent = data.experiments.total_experiments;
            document.getElementById('avgConfidence').textContent = 
                data.experiments.avg_confidence.toFixed(2);
        }
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

// Handle form submission
async function handleSubmit(e) {
    e.preventDefault();
    
    if (isProcessing) return;
    
    const questionInput = document.getElementById('questionInput');
    const question = questionInput.value.trim();
    
    if (!question) return;
    
    // Clear input and hide welcome message
    questionInput.value = '';
    questionInput.style.height = 'auto';
    hideWelcomeMessage();
    
    // Add user message
    addMessage('user', question);
    
    // Show loading state
    const loadingMsg = addLoadingMessage();
    
    // Disable input
    isProcessing = true;
    questionInput.disabled = true;
    document.getElementById('sendBtn').disabled = true;
    
    try {
        // Send research request
        const response = await fetch('/api/research', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });
        
        const data = await response.json();
        
        // Remove loading message
        loadingMsg.remove();
        
        if (data.success) {
            // Add assistant response
            addResearchResult(data);
            
            // Update stats
            totalQueries++;
            document.getElementById('queryCount').textContent = totalQueries;
            loadStats();
        } else {
            addMessage('assistant', `Error: ${data.error}`);
        }
        
    } catch (error) {
        loadingMsg.remove();
        addMessage('assistant', `Error: ${error.message}`);
        console.error('Research failed:', error);
    } finally {
        // Re-enable input
        isProcessing = false;
        questionInput.disabled = false;
        document.getElementById('sendBtn').disabled = false;
        questionInput.focus();
    }
}

// Add message to chat
function addMessage(role, content) {
    const chatMessages = document.getElementById('chatMessages');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.textContent = content;
    
    contentDiv.appendChild(textDiv);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Add loading message
function addLoadingMessage() {
    const chatMessages = document.getElementById('chatMessages');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading';
    loadingDiv.innerHTML = `
        <div class="loading-spinner"></div>
        <span>Researching your question...</span>
    `;
    
    contentDiv.appendChild(loadingDiv);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    
    return messageDiv;
}

// Add research result
function addResearchResult(data) {
    const chatMessages = document.getElementById('chatMessages');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Confidence badge
    const confidence = data.confidence;
    const confidenceClass = confidence >= 0.75 ? 'high' : confidence >= 0.5 ? 'medium' : 'low';
    
    const resultHTML = `
        <div class="research-result">
            <div class="confidence-badge ${confidenceClass}">
                <span>Confidence: ${(confidence * 100).toFixed(0)}%</span>
                <span>${confidence >= 0.75 ? '🎯' : confidence >= 0.5 ? '⚠️' : '❓'}</span>
            </div>
            
            <div class="message-text">${data.answer}</div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Critic Score</div>
                    <div class="metric-value">${(data.metrics.critic_score * 100).toFixed(0)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Consistency</div>
                    <div class="metric-value">${(data.metrics.self_consistency * 100).toFixed(0)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Evidence</div>
                    <div class="metric-value">${(data.metrics.evidence_diversity * 100).toFixed(0)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Iterations</div>
                    <div class="metric-value">${data.iterations}</div>
                </div>
            </div>
            
            ${data.sources.length > 0 ? `
                <div class="sources-section">
                    <div class="section-title">📚 Retrieved Sources</div>
                    ${data.sources.map(source => `
                        <div class="source-item">
                            <div class="source-similarity">Similarity: ${(source.similarity * 100).toFixed(0)}%</div>
                            <div class="source-text">${source.text}</div>
                        </div>
                    `).join('')}
                </div>
            ` : ''}
            
            ${data.critique ? `
                <div class="critique-section">
                    <div class="section-title">🎯 Quality Assessment</div>
                    ${data.critique.strengths.length > 0 ? `
                        <p style="color: var(--success); font-weight: 600; font-size: 0.875rem; margin-bottom: 0.5rem;">Strengths:</p>
                        <ul class="critique-list">
                            ${data.critique.strengths.map(s => `<li>${s}</li>`).join('')}
                        </ul>
                    ` : ''}
                    ${data.critique.weaknesses.length > 0 ? `
                        <p style="color: var(--warning); font-weight: 600; font-size: 0.875rem; margin-top: 1rem; margin-bottom: 0.5rem;">Areas for Improvement:</p>
                        <ul class="critique-list">
                            ${data.critique.weaknesses.map(w => `<li>${w}</li>`).join('')}
                        </ul>
                    ` : ''}
                    <p style="font-size: 0.875rem; color: var(--text-muted); margin-top: 1rem;">
                        Hallucination Risk: <strong style="color: var(--text-secondary);">${data.critique.hallucination_risk}</strong>
                    </p>
                </div>
            ` : ''}
            
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border); font-size: 0.75rem; color: var(--text-muted);">
                Completed in ${data.duration.toFixed(1)}s
            </div>
        </div>
    `;
    
    contentDiv.innerHTML = resultHTML;
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Hide welcome message
function hideWelcomeMessage() {
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.style.display = 'none';
    }
}

// Clear chat
function clearChat() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <h2>Welcome to SR-MARE! 👋</h2>
            <p>Ask me any research question, and I'll provide a comprehensive, evidence-based answer.</p>
            <div class="example-questions">
                <p><strong>Try asking:</strong></p>
                <button class="example-btn" data-question="What are the main causes of climate change?">
                    What are the main causes of climate change?
                </button>
                <button class="example-btn" data-question="How does machine learning work?">
                    How does machine learning work?
                </button>
                <button class="example-btn" data-question="What is quantum entanglement?">
                    What is quantum entanglement?
                </button>
            </div>
        </div>
    `;
    setupEventListeners();
}

// Scroll to bottom
function scrollToBottom() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
