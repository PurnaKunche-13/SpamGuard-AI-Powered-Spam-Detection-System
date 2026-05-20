const apiBase = "http://127.0.0.1:8000";

const button = document.getElementById("analyzeBtn");
const messageField = document.getElementById("message");
const statusBox = document.getElementById("status");
const resultBox = document.getElementById("result");
const labelValue = document.getElementById("labelValue");
const spamValue = document.getElementById("spamValue");
const hamValue = document.getElementById("hamValue");

button.addEventListener("click", async () => {
  const message = messageField.value.trim();
  if (!message) {
    statusBox.textContent = "Enter a message before running analysis.";
    resultBox.classList.add("hidden");
    return;
  }

  button.disabled = true;
  statusBox.textContent = "Analyzing message...";

  try {
    const response = await fetch(`${apiBase}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }

    const label = String(data.label || "").toLowerCase();
    labelValue.textContent = label.toUpperCase();
    labelValue.className = label;
    spamValue.textContent = `${(Number(data.spam_probability) * 100).toFixed(2)}%`;
    hamValue.textContent = `${(Number(data.ham_probability) * 100).toFixed(2)}%`;
    resultBox.classList.remove("hidden");
    statusBox.textContent = "Prediction complete.";
  } catch (error) {
    statusBox.textContent = error.message;
    resultBox.classList.add("hidden");
  } finally {
    button.disabled = false;
  }
});

