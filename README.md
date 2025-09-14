Of course. Based on the sophisticated and well-structured nature of your project, here is a professional, comprehensive `README.md` file written in English.

---

# 🚀 AIC25 Search Fleet: A Hybrid AI Video Search Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

**AIC25 Search Fleet** is an advanced, multi-modal video search engine designed for the AI City Challenge 2025. This project implements a powerful "dual-pronged" strategy, allowing users to query a massive video dataset through two distinct yet complementary lenses:

*   **👁️ Visual Scout:** A semantic and spatial visual search engine. It processes complex natural language queries, decomposing them into searchable visual concepts. It then employs a sophisticated multi-stage reranking pipeline to find the most relevant keyframes.
*   **👂 Transcript Intel:** A high-performance transcript investigation tool. It enables users to perform nested, keyword-based searches across millions of transcript lines, instantly pinpointing moments where specific topics are discussed.

The entire system is wrapped in an intuitive and feature-rich user interface built with Gradio, providing analysts with a powerful cockpit for video exploration, analysis, and submission management.

*(Placeholder for a GIF or screenshot of the application)*


## ✨ Key Features

*   **Hybrid AI Core:** Leverages both **Google Gemini** (for advanced text analysis, query decomposition, and entity grounding) and **OpenAI GPT-4o** (for high-fidelity Visual Question Answering), creating a robust, multi-faceted AI backbone.
*   **"Phoenix" Reranking Pipeline:** A state-of-the-art, multi-stage process for refining search results:
    1.  **Initial Retrieval:** Fast candidate selection using a FAISS index with CLIP embeddings.
    2.  **Semantic Reranking:** Refines scores using a Vietnamese-specific Bi-Encoder model.
    3.  **Spatial Filtering:** Verifies spatial relationships between detected objects (e.g., "a person *behind* a car").
    4.  **Fine-Grained Verification:** Uses CLIP to analyze cropped object regions for detailed attributes (e.g., "a flag with a yellow star").
*   **Dynamic Query Decomposer:** Intelligently breaks down long, complex user queries into multiple, simpler, and visually searchable sub-queries, which are then executed in parallel.
*   **Result Diversity Engine:** Implements Maximal Marginal Relevance (MMR) to prevent result duplication and ensure the final output is diverse and comprehensive.
*   **Advanced Transcriptomics:** A highly optimized transcript search engine that allows for chained filtering, keyword highlighting, and seamless integration with the visual analysis panel.
*   **Comprehensive Interactive UI:**
    *   Dual-tab interface for clear separation between visual and transcript search.
    *   A unified analysis panel showing the selected keyframe, a 30-second video preview, the full video transcript, and a detailed scoring breakdown.
    *   Integrated full-video player that copies source files on-demand to avoid breaking Gradio's file access rules.
*   **Robust Submission Workflow:**
    *   Easily add candidates from either search method to a persistent submission list.
    *   A live-editable submission editor that syncs with the internal list.
    *   Automated frame index calculation based on video-specific FPS maps.
    *   One-click generation of properly formatted `.csv` submission files.

## 🏗️ System Architecture

The project is logically divided into three main layers: Frontend (UI), Backend (Logic & Data Loading), and the Search Core (AI Engines).

1.  **Frontend (Gradio Interface)**
    *   `app.py`: The main entry point that launches the Gradio server.
    *   `ui_layout.py`: Defines the complete layout and structure of all UI components.
    *   `ui_helpers.py`: Provides helper functions for generating dynamic HTML content (like the score analysis panel) and formatting data for display.
    *   `event_handlers.py`: The "central nervous system" of the UI, connecting user actions (button clicks, selections) to the appropriate backend functions.

2.  **Backend (Orchestration & Data Loading)**
    *   `backend_loader.py`: Responsible for initializing all core backend components. It loads AI models, reads large data files (metadata, FAISS index), and maps all video paths across multiple data batches.
    *   `config.py`: A centralized configuration file for all file paths, constants, and API keys, which are securely loaded from Kaggle Secrets.

3.  **Search Core (The Engine Room)**
    *   `master_searcher.py`: The primary orchestrator. It receives the user query, determines the task type, coordinates the various sub-searchers and AI handlers, and returns the final, polished result.
    *   `basic_searcher.py`: Implements the initial, fast retrieval stage using a FAISS index and a CLIP model for encoding text queries.
    *   `semantic_searcher.py`: Contains the logic for the multi-stage "Phoenix" reranking pipeline.
    *   `query_decomposer.py`: A dedicated class using Gemini to break down complex queries.
    *   `gemini_text_handler.py` & `openai_handler.py`: Specialized handlers that act as adapters to the Google and OpenAI APIs, encapsulating logic for API calls, retries, and error handling.
    *   `task_analyzer.py`: Classifies the user's intent into different task types (e.g., simple KIS, Q&A, or Action Tracking).
    *   `mmr_builder.py`: Implements the MMR algorithm to diversify search results.
    *   `transcript_searcher.py`: A high-performance, in-memory search engine for video transcripts.
    *   `utils/`: A package of helper modules for common tasks like video clipping (`video_utils.py`), spatial relationship logic (`spatial_engine.py`), and submission formatting (`formatting.py`).

## 🚀 Setup and Launch

1.  **Clone the Repository:**
    ```bash
    git clone https://your-repository-url/Project_AIC_Ver2.git
    cd Project_AIC_Ver2
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Keys:**
    *   Navigate to your Kaggle notebook's "Secrets" section.
    *   Add your API keys with the following exact names:
        *   `OPENAI_API_KEY`
        *   `GOOGLE_API_KEY` (for the Gemini API)
    *   The `config.py` file is pre-configured to load these secrets automatically.

4.  **Verify Data Paths:**
    *   Open `config.py`.
    *   Ensure that all paths (`CLIP_FEATURES_PATH`, `FAISS_INDEX_PATH`, etc.) correctly point to your dataset files within the `/kaggle/input/` directory.
    *   Update the `VIDEO_BASE_PATHS` and `KEYFRAME_BASE_PATHS` lists to include the paths for all data batches you intend to use.

5.  **Run the Application:**
    ```bash
    python app.py
    ```
    The script will initialize all backend components and launch the Gradio web server. A public-sharing URL will be generated for you to access the interface.

## 📂 Project Structure

```
Project_AIC_Ver2/
 ┣ search_core/
 ┃ ┣ basic_searcher.py         # Core retrieval engine (FAISS + CLIP)
 ┃ ┣ gemini_text_handler.py    # Handler for Google Gemini API
 ┃ ┣ master_searcher.py        # Main search orchestrator
 ┃ ┣ mmr_builder.py            # Result diversification engine (MMR)
 ┃ ┣ openai_handler.py         # Handler for OpenAI API (VQA)
 ┃ ┣ query_decomposer.py       # Decomposes complex queries
 ┃ ┣ semantic_searcher.py      # Multi-stage reranking engine
 ┃ ┣ task_analyzer.py          # Classifies query types (KIS, QNA, TRAKE)
 ┃ ┣ trake_solver.py           # Solves action sequence tracking tasks
 ┃ ┣ transcript_searcher.py    # High-performance transcript search
 ┃ ┗ vqa_handler.py            # Visual Question Answering logic
 ┣ utils/
 ┃ ┣ api_utils.py              # Decorator for API call retries
 ┃ ┣ cache_manager.py          # Caches computed object vectors
 ┃ ┣ formatting.py             # Formats data for submission and UI
 ┃ ┣ image_cropper.py          # Crops image regions by bounding box
 ┃ ┣ spatial_engine.py         # Logic for spatial relationship checks
 ┃ ┣ video_utils.py            # Video processing utilities (e.g., clipping)
 ┃ ┗ __init__.py
 ┣ app.py                      # Main Gradio application entry point
 ┣ backend_loader.py           # Initializes all backend components
 ┣ config.py                   # Central configuration for paths and keys
 ┣ event_handlers.py           # Functions triggered by UI events
 ┣ README.md                   # This documentation file
 ┣ requirements.txt            # Python dependencies
 ┣ ui_helpers.py               # Helper functions for building the UI
 ┗ ui_layout.py                # Defines the Gradio UI component layout
```

## 📜 License

This project is distributed under the MIT License. See the `LICENSE` file for more information.