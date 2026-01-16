document.addEventListener("DOMContentLoaded", () => {
    // Section Elements 
    const analysisSection = document.getElementById('analysis-section');
    const resultsSection = document.getElementById('results-section');
    const chatContainer = document.getElementById('chat-container');
    
    // Analysis Elements 
    const uploadForm = document.getElementById('upload-form');
    const analyzeButton = document.getElementById('analyze-button');
    const analysisSpinner = document.getElementById('analysis-spinner');
    const analysisError = document.getElementById('analysis-error');
    
    // Results Elements
    const actualImage = document.getElementById('actual-image');
    const predictedImage = document.getElementById('predicted-image');
    const downloadCaptionBtn = document.getElementById('download-caption-btn');
    const downloadPdfBtn = document.getElementById('download-pdf-btn');
    const medicalReportContent = document.getElementById('medical-report-content');
    const groundingCaptionContent = document.getElementById('grounding-caption-content');
    const tabs = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    // Chat Elements 
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");

    let chatHistory = [];
    let actualPanzoom = null;
    let predictedPanzoom = null;

    // ANALYSIS LOGIC 
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        analysisSpinner.style.display = 'block';
        analysisError.style.display = 'none';
        analysisError.innerText = '';
        analyzeButton.disabled = true;
        analyzeButton.innerText = 'Analyzing...';

        const formData = new FormData(uploadForm);
        
        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || `Server error: ${response.status}`);
            }

            // Success: Populate Results
            // Add cache-buster to image URLs to force reload
            actualImage.src = result.original_image_url + '?t=' + new Date().getTime();
            predictedImage.src = result.predicted_image_url + '?t=' + new Date().getTime();
            
            if (actualPanzoom) actualPanzoom.destroy();
            if (predictedPanzoom) predictedPanzoom.destroy();

            actualPanzoom = Panzoom(actualImage, {
                maxScale: 10, // Allow zooming in 10x
                contain: 'outside'
            });
            actualImage.parentElement.addEventListener('wheel', actualPanzoom.zoomWithWheel);

            predictedPanzoom = Panzoom(predictedImage, {
                maxScale: 10, // Allow zooming in 10x
                contain: 'outside'
            });
            predictedImage.parentElement.addEventListener('wheel', predictedPanzoom.zoomWithWheel);

            // Use marked.parse for the report, but not for the caption
            medicalReportContent.innerHTML = marked.parse(result.report);
            groundingCaptionContent.innerText = result.caption;

            // Show results and chat
            analysisSection.style.display = 'none';
            resultsSection.style.display = 'block';
            chatContainer.style.display = 'flex';

            // Start the chat
            startChat();

        } catch (error) {
            analysisError.innerText = `Analysis Failed: ${error.message}`;
            analysisError.style.display = 'block';
        } finally {
            analysisSpinner.style.display = 'none';
            analyzeButton.disabled = false;
            analyzeButton.innerText = 'Analyze Image';
        }
    });

    // RESULTS & DOWNLOAD LOGIC
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(tab.dataset.tab).classList.add('active');
        });
    });

    downloadCaptionBtn.addEventListener('click', () => {
        window.location.href = '/download_caption';
    });

    downloadPdfBtn.addEventListener('click', () => {
        // The /download_pdf route will handle the redirect
        window.open('/download_pdf', '_blank');
    });


    // CHAT LOGIC
    const addMessage = (sender, message, isHtml = false) => {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", sender);
        
        if (isHtml) {
            messageDiv.innerHTML = message;
        } else {
            messageDiv.innerText = message;
        }
        
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        return messageDiv;
    };

    const startChat = () => {
        chatHistory = [];
        const greeting = "Hello, I have your dental report here and can help answer any questions you have about it.";
        addMessage("bot", marked.parse(greeting), true);
    };

    const handleSend = async () => {
        const question = userInput.value.trim();
        if (!question) return;

        addMessage("user", question);
        
        const currentHistory = [...chatHistory];
        chatHistory.push({ "role": "user", "content": question });
        userInput.value = "";
        
        sendButton.disabled = true;
        const botMessageDiv = addMessage("bot", "");
        botMessageDiv.classList.add("loading");

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: question,
                    history: currentHistory
                })
            });

            if (!response.ok) {
                const err = await response.json();
                botMessageDiv.innerText = `Error: ${err.error || response.statusText}`;
                botMessageDiv.classList.remove("loading");
                botMessageDiv.classList.add("error");
                chatHistory.pop(); 
                return;
            }
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullResponse = "";
            botMessageDiv.classList.remove("loading");

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                
                if (chunk.startsWith("ERROR:")) {
                    fullResponse = chunk;
                    botMessageDiv.innerHTML = marked.parse(fullResponse);
                    botMessageDiv.classList.add("error");
                    break;
                } else {
                    fullResponse += chunk;
                    botMessageDiv.innerHTML = marked.parse(fullResponse); 
                }
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            if (fullResponse.startsWith("ERROR:")) {
                 botMessageDiv.classList.add("error");
                 chatHistory.pop();
            } else {
                chatHistory.push({ "role": "assistant", "content": fullResponse });
            }

        } catch (error) {
            botMessageDiv.innerText = `Error: ${error.message}`;
            botMessageDiv.classList.remove("loading");
            botMessageDiv.classList.add("error");
            chatHistory.pop(); 
        } finally {
            sendButton.disabled = false;
            userInput.focus();
        }
    };

    sendButton.addEventListener("click", () => handleSend(false));
    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            handleSend(false);
        }
    });

    // Feedback Modal Logic

    const feedbackBtn = document.getElementById('feedback-btn');
    const feedbackModal = document.getElementById('feedback-modal');
    const feedbackOverlay = document.getElementById('feedback-overlay');
    const feedbackForm = document.getElementById('feedback-form');
    const cancelFeedbackBtn = document.getElementById('cancel-feedback-btn');
    const submitFeedbackBtn = document.getElementById('submit-feedback-btn');
    const feedbackName = document.getElementById('feedback-name');
    const feedbackText = document.getElementById('feedback-text');
    const feedbackStatus = document.getElementById('feedback-status');

    // Function to show the modal
    function showFeedbackModal() {
        feedbackModal.classList.remove('hidden');
        // Clear old status messages
        feedbackStatus.classList.add('hidden');
        feedbackStatus.textContent = '';
        submitFeedbackBtn.disabled = false;
    }

    // Function to hide the modal
    function hideFeedbackModal() {
        feedbackModal.classList.add('hidden');
        feedbackForm.reset(); // Clear the form
    }

    // Show modal when "Give Feedback" is clicked
    feedbackBtn.addEventListener('click', showFeedbackModal);
    
    // Hide modal on cancel or overlay click
    cancelFeedbackBtn.addEventListener('click', hideFeedbackModal);
    feedbackOverlay.addEventListener('click', hideFeedbackModal);

    // Handle the feedback form submission
    feedbackForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        submitFeedbackBtn.disabled = true;
        
        const name = feedbackName.value || 'Anonymous'; 
        const feedback_text = feedbackText.value;

        if (!feedback_text) {
            showFeedbackStatus('Please enter your feedback.', 'error');
            submitFeedbackBtn.disabled = false;
            return;
        }

        const payload = {
            name: name,
            feedback_text: feedback_text,
            source: 'website'
        };

        try {
            const response = await fetch('/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'Failed to submit feedback.');
            }
            
            // Success!
            showFeedbackStatus('Thank you! Your feedback has been submitted.', 'success');
            setTimeout(hideFeedbackModal, 2000);

        } catch (error) {
            showFeedbackStatus(error.message, 'error');
            submitFeedbackBtn.disabled = false;
        }
    });

    function showFeedbackStatus(message, type) {
        feedbackStatus.textContent = message;
        if (type === 'success') {
            feedbackStatus.className = 'feedback-status-message success';
        } else { 
            feedbackStatus.className = 'feedback-status-message error';
        }
        feedbackStatus.classList.remove('hidden');
    }

});