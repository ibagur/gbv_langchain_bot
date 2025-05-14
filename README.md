---
title: GBV Langchain Bot
emoji: 🐢
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 3.36.1
app_file: app.py
pinned: false
---

# GBV in Emergencies Q&A Bot

## Overview

This is a question-answering chatbot specialized in Gender-Based Violence (GBV) response in humanitarian settings. The bot uses LangChain and OpenAI's language models to provide accurate information by drawing from a specialized knowledge base while supplementing answers with Wikipedia data when necessary.

## Features

- Interactive chat interface built with Gradio
- Retrieval-augmented generation using a vector database (Chroma)
- Conversational memory with summarization capability
- Multiple knowledge sources (specialized GBV documents + Wikipedia)
- Streaming responses for better user experience

## Architecture

The application uses several key components:

1. **Vector Database**: Chroma DB storing embeddings of GBV-related documents
2. **Language Model**: OpenAI's GPT models (configurable between GPT-3.5-Turbo and GPT-4)
3. **Retrieval System**: LangChain's RetrievalQAWithSourcesChain for accurate information retrieval
4. **Agent Framework**: Conversational ReAct agent to intelligently route between knowledge sources
5. **Memory System**: ConversationSummaryBufferMemory to maintain context in long conversations

## Usage

Users can ask questions related to GBV in humanitarian settings. Example questions:
- What are the GBV guiding principles?
- Which UN agency leads the GBV response in emergencies?
- How can we engage men and boys in GBV prevention and response?
- Please outline a strategy to minimize GBV risks in a temporary settlement
- What is the integration factor between GBV and SRH?

## Setup and Deployment

1. Ensure you have Python installed
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your OpenAI API key as an environment variable: `OPENAI_TOKEN`
4. Run the application: `python app.py`
5. Access the interface through your browser

## Deployment on Hugging Face Spaces

This application is configured for deployment on Hugging Face Spaces using the Gradio SDK. The configuration is defined in the frontmatter of this README.

## License

This project is intended for educational and humanitarian purposes. Please use responsibly.
