This project implements a chatbot application that leverages a local Large Language Model (LLM) using [Ollama](https://ollama.com/) and integrates a hybrid search functionality powered by [Langchain](https://www.langchain.com/) and the [Serper API](https://serper.dev/).

The application features both a backend API (built with FastAPI) for handling chat and search logic, and a frontend user interface (built with Streamlit) for interaction.

Features
-Local LLM Integration : Uses Ollama to run large language models locally on your machine (e.g., `tinyllama:chat`).
-Hybrid Chat Mode : Combines the power of the local LLM with real-time web search results (via Serper API) to provide more accurate and up-to-date answers.
-Web Search Mode : Allows for direct web searches with sources.
-Streamlit Frontend : An interactive and user-friendly web interface for the chatbot.
-FastAPI Backend : A robust and scalable API to handle requests from the frontend.
-Environment Variable Management : Utilizes `.env` files for secure API key management.

ensure you have the following installed:

Python 3.11+ :This project was developed with Python 3.11 in mind, ideally within a Conda virtual environment.
Conda (Recommended) : For managing Python environments and dependencies. [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html).
Ollama: The Ollama desktop application for macOS (or your respective OS) must be installed and running. [Download Ollama](https://ollama.com/).
Serper API Key :You will need an API key from Serper for the web search functionality. [Get your API key here](https://serper.dev/).


Setup and Installation

Follow these steps to get the project up and running on your local machine.

1. Clone the Repository

First, clone this GitHub repository to your local machine:

```bash
$git clone https://github.com/ManojKurapati/avirock_proj
$cd avirock_proj
$conda create -n landonorris python=3.11
$conda activate landonorris
$pip install -r requirements.txt
$ollama serve
$uvicorn server:app --reload  Or $Python -m uvicorn server:app --reload
$streamlit run client.py


API_Endpoints

Endpoint	                            Purpose
POST /chat/llm_only	                    Pure LLM response (streaming)
GET /search?q=...	                    Search-only mode with summary
POST /chat/hybrid	                    Auto LLM + web hybrid logic
POST /v1/chat/completions	            OpenAI-style JSON schema interface
