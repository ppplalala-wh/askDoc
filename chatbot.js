async function sendMessageToBackend(userInput) {
  try {
    const response = await fetch('http://localhost:8087/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ message: userInput })
    });

    const data = await response.json();
    return data.message;
  } catch (error) {
    console.log('Error:', error);
    return 'Oops! Something went wrong.';
  }
}

function appendMessage(message, className) {
  const chatWindow = document.getElementById('chat-window');
  const messageElement = document.createElement('div');
  messageElement.classList.add('message');
  messageElement.classList.add(className);
  messageElement.textContent = message;
  chatWindow.appendChild(messageElement);
}

document.addEventListener('DOMContentLoaded', () => {
  // Your code here
    document.getElementById('chat-form').addEventListener('submit', async (event) => {
      event.preventDefault();
      const userInput = document.getElementById('message-input').value;
      appendMessage(userInput, 'user-message'); // Display user message in chat window
      document.getElementById('message-input').value = ''; // Clear the input field

      const botResponse = await sendMessageToBackend(userInput);
      appendMessage(botResponse, 'bot-message'); // Display bot response in chat window
    });
});







