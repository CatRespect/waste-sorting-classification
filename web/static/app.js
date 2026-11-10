const input = document.querySelector("#image-input");
const zone = document.querySelector("#drop-zone");
const preview = document.querySelector("#preview");
const emptyState = document.querySelector("#empty-state");
const fileName = document.querySelector("#file-name");
const submit = document.querySelector("#submit-button");
const status = document.querySelector("#status");
const result = document.querySelector("#result");
const resultEmpty = document.querySelector("#result-empty");
let selectedFile;

function showFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    status.textContent = "Please choose an image file.";
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    status.textContent = "The image must be no larger than 10 MB.";
    return;
  }
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  emptyState.hidden = true;
  fileName.textContent = file.name;
  fileName.hidden = false;
  submit.disabled = false;
  status.textContent = "";
}

input.addEventListener("change", () => showFile(input.files[0]));
["dragenter", "dragover"].forEach((event) => zone.addEventListener(event, (e) => {
  e.preventDefault(); zone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((event) => zone.addEventListener(event, (e) => {
  e.preventDefault(); zone.classList.remove("dragging");
}));
zone.addEventListener("drop", (event) => showFile(event.dataTransfer.files[0]));

submit.addEventListener("click", async () => {
  if (!selectedFile) return;
  const form = new FormData();
  form.append("image", selectedFile);
  submit.disabled = true;
  submit.classList.add("loading");
  status.textContent = "The model is analysing the image…";
  try {
    const response = await fetch("/api/predict", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Recognition could not be completed.");
    document.querySelector("#result-label").textContent = data.prediction.label;
    document.querySelector("#result-confidence").textContent = `${data.prediction.confidence}% confidence`;
    document.querySelector("#confidence-bar").style.width = `${data.prediction.confidence}%`;
    document.querySelector("#alternatives").replaceChildren(...data.alternatives.map((item) => {
      const li = document.createElement("li");
      li.textContent = `${item.label} — ${item.confidence}%`;
      return li;
    }));
    resultEmpty.hidden = true;
    result.hidden = false;
    status.textContent = "Done.";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    submit.disabled = false;
    submit.classList.remove("loading");
  }
});
