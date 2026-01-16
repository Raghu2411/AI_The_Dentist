document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");
    const patientIdInput = document.getElementById("patient-id");

    let chatHistory = [];

    const addMessage = (sender, message) => {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", sender);
        messageDiv.innerText = message;
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        return messageDiv;
    };

    const handleSend = async () => {
        const question = userInput.value.trim();
        const patient_pdf_name = patientIdInput.value.trim();

        if (!question) return;
        if (!patient_pdf_name) {
            addMessage("bot", "ERROR: Please enter a Patient File ID above.");
            return;
        }

        addMessage("user", question);
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
                    history: chatHistory,
                    patient_id: patient_pdf_name // Send the full PDF name
                })
            });

            if (!response.ok) {
                const err = await response.json();
                botMessageDiv.innerText = `Error: ${err.error || response.statusText}`;
                botMessageDiv.classList.remove("loading");
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
                fullResponse += chunk;
                botMessageDiv.innerHTML = marked.parse(fullResponse); 
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            if (fullResponse.startsWith("ERROR:")) {
                 botMessageDiv.classList.add("error");
            } else {
                chatHistory.push({ "role": "user", "content": question });
                chatHistory.push({ "role": "assistant", "content": fullResponse });
            }

        } catch (error) {
            botMessageDiv.innerText = `Error: ${error.message}`;
            botMessageDiv.classList.remove("loading");
            botMessageDiv.classList.add("error");
        } finally {
            sendButton.disabled = false;
            userInput.focus();
        }
    };

    sendButton.addEventListener("click", handleSend);
    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            handleSend();
        }
    });
});