// Configure marked.js for markdown parsing
marked.setOptions({
    breaks: true,
    gfm: true
});

// Configure MathJax for math equations
window.MathJax = {
    tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        processEscapes: true
    },
    options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
    }
};

const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
let isProcessing = false; // Prevent double submissions

async function sendMessage() {
    console.log('sendMessage called, isProcessing:', isProcessing);
    
    // Prevent double submissions
    if (isProcessing) {
        console.log('Already processing, ignoring...');
        return;
    }

    const question = userInput.value.trim();
    if (!question) return;
    
    // Lock to prevent double submissions
    isProcessing = true;
    
    // Add user message to chat
    appendMessage("user", `You: ${question}`);
    userInput.value = "";
    
    // Add loading indicator
    const loadingId = appendMessage("bot", "🤔 Thinking...", false, true);
    
    try {
        console.log('Sending request to server...');
        
        // Call FastAPI backend
        const response = await fetch("http://localhost:8000/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question })
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Received data:', data);
        
        // Remove loading indicator
        removeMessage(loadingId);
        
        // Add bot response with markdown parsing
        appendMessage("bot", `Bot: ${data.answer}`, true);
        
        if (data.source) {
            appendMessage("source-msg", `📚 Source: ${data.source}`);
        }
        
    } catch (error) {
        console.error('Fetch error:', error);
        
        // Remove loading indicator
        removeMessage(loadingId);
        
        // Add error message
        appendMessage("bot", "Bot: ❌ Error - Could not get response");
        
    } finally {
        // Always unlock, even if there's an error
        isProcessing = false;
        console.log('Process completed, unlocked');
    }
}

function appendMessage(sender, text, isMarkdown = false, isTemporary = false) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `${sender}${sender.endsWith('-msg') ? '' : '-msg'}`;
    
    // Generate unique ID for temporary messages
    if (isTemporary) {
        msgDiv.id = `temp-${Date.now()}-${Math.random()}`;
    }
    
    if (isMarkdown && sender === "bot") {
        // Parse markdown and render
        msgDiv.innerHTML = marked.parse(text);
        // Process math after adding to DOM
        setTimeout(() => {
            if (window.MathJax) {
                MathJax.typesetPromise([msgDiv]).catch(err => console.log('MathJax error:', err));
            }
        }, 100);
    } else {
        msgDiv.textContent = text;
    }
    
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    
    // Return ID for temporary messages
    return isTemporary ? msgDiv.id : null;
}

function removeMessage(messageId) {
    if (messageId) {
        const msgDiv = document.getElementById(messageId);
        if (msgDiv) {
            msgDiv.remove();
        }
    }
}

// Event listeners - IMPORTANT: Only add them once when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, adding event listeners...');
    
    // Button click event
    const sendButton = document.querySelector("button");
    if (sendButton) {
        sendButton.addEventListener("click", function(e) {
            e.preventDefault();
            e.stopPropagation();
            console.log('Button clicked');
            sendMessage();
        });
    }
    
    // Enter key event
    if (userInput) {
        userInput.addEventListener("keydown", function(e) {
            if (e.key === "Enter" && !isProcessing) {
                e.preventDefault();
                e.stopPropagation();
                console.log('Enter pressed');
                sendMessage();
            }
        });
        
        // Focus input
        userInput.focus();
    }
});