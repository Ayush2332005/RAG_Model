# 🎬 Video-to-RAG System (Whisper + Ollama)

## 📌 Overview

This project builds a complete **Video Understanding System** using RAG (Retrieval-Augmented Generation).

It:

* Converts 🎥 videos → 🎧 audio
* Transcribes audio → 📝 text using Whisper
* Processes text into chunks
* Generates embeddings using Ollama
* Performs semantic search
* Answers user queries with timestamps

---

## 🧱 Project Structure

```bash
RAG/
│
├── videos/                # Input video files
├── audios/                # Extracted audio files
├── jsons/
│   └── chunks.json        # Processed subtitle chunks
│
├── video_to_mp3.py        # Convert video → audio
├── mp3_to_json.py         # Whisper transcription
├── preprocess_json.py     # Clean + chunk text
├── embedding.py           # Create embeddings
├── process_incoming.py    # Query + retrieval
│
├── embeddings.joblib      # Stored embeddings
├── file_list.txt          # List of video files
├── prompt.txt             # Prompt template
├── response.txt           # Output response
│
├── venv/                  # Virtual environment
└── README.md
```

---

## 🚀 Pipeline

```text
Video → Audio → Whisper → JSON → Chunking → Embeddings → Search → Answer
```

---

## ⚙️ Setup Instructions

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 2️⃣ Install Dependencies

```bash
pip install openai-whisper
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn joblib requests
pip install ffmpeg-python
```

---

### 3️⃣ Install FFmpeg (IMPORTANT)

* Download: https://ffmpeg.org/download.html
* Add to PATH

Verify:

```bash
ffmpeg -version
```

---

### 4️⃣ Install Ollama

Download: https://ollama.com

Run:

```bash
ollama serve
```

---

### 5️⃣ Pull Models

#### 🔹 Embedding Model

```bash
ollama pull nomic-embed-text
```

#### 🔹 LLM (recommended)

```bash
ollama pull mistral
```

---

## 🧩 Step-by-Step Execution

---

### 🔹 Step 1: Add Video Files

Update `file_list.txt`:

```text
file 'videos/advanced_intro.mp4'
file 'videos/intermediate.mp4'
```

---

### 🔹 Step 2: Convert Video → Audio

```bash
python video_to_mp3.py
```

---

### 🔹 Step 3: Audio → Text (Whisper)

```bash
python mp3_to_json.py
```

Output:

```text
jsons/chunks.json
```

---

### 🔹 Step 4: Preprocess + Chunk

```bash
python preprocess_json.py
```

---

### 🔹 Step 5: Generate Embeddings

```bash
python embedding.py
```

Output:

```text
embeddings.joblib
```

---

### 🔹 Step 6: Query System

```bash
python process_incoming.py
```

Example:

```text
Enter your query: What is chunking?
```

---

## 🔍 How It Works

### 1. 🎙️ Transcription

* Uses Whisper to convert speech → text

---

### 2. ✂️ Chunking

* Splits transcript into meaningful parts
* Each chunk contains:

  * text
  * start time
  * end time

---

### 3. 🧠 Embeddings

* Uses:

```text
nomic-embed-text
```

* Converts text → vectors

---

### 4. 🔎 Retrieval

* Uses cosine similarity
* Finds most relevant chunks

---

### 5. 🤖 Answer Generation

* Sends chunks to LLM
* Returns:

  * Answer
  * Video reference
  * Timestamp

---

## 🧠 Example Output

```text
📹 Video: advanced_intro.mp4  
⏱️ Timestamp: 72s – 78s  
📝 Chunking is explained as combining words into meaningful phrases.
```

---

## ⚠️ Common Issues & Fixes

### ❌ File not found

✔ Check path:

```python
os.path.exists(path)
```

---

### ❌ Ollama embedding error

✔ Use:

```python
data["embedding"]
```

---

### ❌ Low RAM error

```text
model requires more memory
```

✔ Use smaller model:

```bash
ollama pull mistral
```

---

### ❌ JSON issues

✔ If list:

```python
for c in content
```

---

## 💡 Improvements (Future Work)

* 🔥 Replace cosine similarity with FAISS
* 🔥 Add Streamlit UI
* 🔥 Multi-video search
* 🔥 Better chunk filtering (remove "Okay")
* 🔥 Batch embeddings (faster)

---

## 🏁 Conclusion

This project demonstrates a **real-world RAG system**:

* Video understanding
* Semantic search
* Context-aware answering

---

## 👨‍💻 Author

**Ayush Mehar**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
