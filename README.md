
# Advanced Multimodal RAG with ColPali and Qwen-VL

This project implements an advanced Retrieval-Augmented Generation (RAG) system that goes beyond simple text extraction. It leverages multimodal models to visually understand the structure, layout, tables, and figures within PDF documents, providing more accurate and context-aware answers.

Traditional RAG pipelines often fail with complex documents by flattening them into plain text, losing critical layout information. This project uses **ColPali** to create visual embeddings that preserve this structure and **Qwen2.5-VL**, a powerful open-source vision-language model, to generate answers based on the actual appearance of the document pages.

---

## 🏛️ Architecture

The system is composed of two main pipelines: Ingestion and Inference.

### 1. Ingestion Pipeline:
- **PDF to Image Conversion:** Each page of an input PDF is converted into a high-resolution image using `pdf2image`.
- **Visual Embedding:** The **ColPali** model (`vidore/colpali`) processes each page image to generate multi-vector embeddings. These embeddings capture not just the text but also its spatial relationships, formatting, and layout.
- **Vector Storage:** The generated visual embeddings are stored in a **Pinecone** vector index for efficient, large-scale similarity search.

### 2. Inference (Query) Pipeline:
- **Query Embedding:** The user's text query is embedded using the same ColPali model to ensure consistency.
- **Visual Retrieval:** The query vector is used to search the Pinecone index, retrieving the page images that are most visually and semantically relevant to the query.
- **Multimodal Generation:** The retrieved page image(s) and the original query are passed to the **Qwen2.5-VL-7B-Instruct** model. This powerful vision-language model analyzes the visual evidence directly to generate a final, accurate answer.

---

## ✨ Features

- **True Visual Understanding:** Processes documents as images, preserving tables, charts, and layout.
- **Layout-Aware Retrieval:** Finds relevant pages based on visual and semantic similarity, not just keywords.
- **Fully Open-Source Generation:** Uses a powerful, locally-runnable open-source model (Qwen2.5-VL) for the final answer generation, avoiding cloud API content restrictions.
- **Modular Design:** Components for embedding, storage, and generation can be swapped out.
- **End-to-End Pipeline:** Provides a complete, runnable Jupyter notebook from PDF ingestion to final query.

---

## 🛠️ Tech Stack

- **Embedding Model:** ColPali (`colpali_engine==0.1.1`)
- **Generator Model:** Qwen/Qwen2.5-VL-7B-Instruct
- **Core Libraries:** PyTorch, Hugging Face `transformers`
- **Vector Database:** Pinecone
- **PDF Processing:** `pdf2image` & `poppler`

---

## 🚀 Setup and Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. System Dependencies

This project uses `pdf2image`, which requires the `poppler` utility.

**On Debian/Ubuntu:**
```bash
sudo apt-get update && sudo apt-get install -y poppler-utils
```

**On macOS (using Homebrew):**
```bash
brew install poppler
```

### 3. Python Environment & Dependencies

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install all required Python packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

#### `requirements.txt` content:
```
pinecone-client==4.1.1
colpali_engine==0.1.1
git+https://github.com/huggingface/transformers
accelerate
qwen-vl-utils[decord]==0.0.8
bitsandbytes
pdf2image
python-dotenv
Pillow
torch
ipython
```

### 4. API Keys Configuration

This project requires an API key from Pinecone for the vector database.

1. Create a `.env` file in the root of the project directory.
2. Add your Pinecone API key:

```env
# .env
PINECONE_API_KEY="YOUR_PINECONE_API_KEY_HERE"
```

---

## ▶️ Usage

The entire pipeline is contained within the `multi_modal_rag.ipynb` Jupyter notebook.

### Steps:

1. **Launch Jupyter:**
    ```bash
    jupyter notebook multi_modal_rag.ipynb
    ```

2. **Run the Cells Sequentially:**
    - **Cell 1: Installations** – Installs all dependencies.  
      ⚠️ If using Colab, restart runtime after this cell finishes.
    - **Cell 2: API Keys** – Loads Pinecone key from `.env`.
    - **Cell 3: PDF Upload** – Upload the document you want to query.
    - **Cells 4–5: Model Loading** – Loads ColPali and Qwen2.5-VL-7B models.
    - **Cells 6–7: Ingestion** – Converts PDF to images, creates embeddings, and stores them in Pinecone.
    - **Cells 8–9: Querying** – Run queries and get layout-aware answers from the retrieved images.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙏 Acknowledgments

- The creators of the **ColPali** methodology for their innovative work in visual document retrieval.
- The **Qwen Team** for developing and open-sourcing the powerful Qwen2.5-VL models.
