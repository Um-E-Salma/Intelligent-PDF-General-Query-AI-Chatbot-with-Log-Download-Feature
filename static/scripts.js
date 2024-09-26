document.addEventListener('DOMContentLoaded', function() {
    const pdfViewer = document.getElementById('pdf-viewer');
    const pdfInput = document.getElementById('pdf-input');
    const processBtn = document.getElementById('process-btn');
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const downloadLogsBtn = document.getElementById('download-logs-btn');
    let logs = [];

    // Function to process the selected PDF
    function processPdf() {
        if (pdfInput.files.length > 0) {
            const selectedPdf = pdfInput.files[0];
            const formData = new FormData();
            formData.append('pdf', selectedPdf);

            fetch('/upload_pdf', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.message) {
                    alert(data.message);

                    // Display the PDF
                    const reader = new FileReader();
                    reader.onload = function(event) {
                        pdfViewer.setAttribute('src', event.target.result);
                    }
                    reader.readAsDataURL(selectedPdf);
                } else if (data.error) {
                    alert(data.error);
                }
            })
            .catch(error => console.error('Error:', error));
        } else {
            alert('Please select a PDF file.');
        }
    }

    // Event listener for the process button
    processBtn.addEventListener('click', function() {
        processPdf();
    });

    // Function to handle user input and send it to the backend
    function handleUserInput() {
        const message = userInput.value.trim().toLowerCase();

        // Display user message in the chat
        displayMessage('You: ' + message, 'user');

        // Send user message to the backend
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ messages: [{ role: 'user', content: message }] })
        })
        .then(response => response.json())
        .then(data => {
            const botResponse = data.content || data.error;
            // Display the message content in the chat window
            displayMessage('Chatbot: ' + botResponse, 'bot');
            // Log the message
            logs.push({ user: message, bot: botResponse });
        })
        .catch(error => console.error('Error:', error));

        userInput.value = '';
    }

    // Function to display messages in the chat
    function displayMessage(message, type) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', type === 'user' ? 'user-message' : 'bot-message');
        messageElement.innerText = message;
        chatMessages.appendChild(messageElement);
    }

    // Event listener for user input
    userInput.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            handleUserInput();
        }
    });

    // Event listener for send button
    sendBtn.addEventListener('click', function() {
        handleUserInput();
    });

    // Function to download logs as JSON
    function downloadLogs() {
        const blob = new Blob([JSON.stringify(logs, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'logs.json';
        a.click();
        URL.revokeObjectURL(url);
    }

    // Event listener for download logs button
    downloadLogsBtn.addEventListener('click', function() {
        downloadLogs();
    });
});
