async function translateText() {
  const text = document.getElementById('inputText').value;
  const targetLanguage = document.getElementById('languageSelect').value;
  const outputDiv = document.getElementById('output');

  if (!text) {
    alert("Please enter some text.");
    return;
  }

  outputDiv.innerText = "⏳ Translating...";

  try {
    const response = await fetch('http://localhost:8000/translate/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, target_language: targetLanguage })
    });

    if (!response.ok) throw new Error("Server error");

    const data = await response.json();
    outputDiv.innerText = data.translated_text || "⚠️ Translation failed.";
  } catch (error) {
    outputDiv.innerText = "❌ Error connecting to server.";
  }
}
