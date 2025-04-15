# MVP Startup Assistant

This project is an AI-powered MVP (Minimum Viable Product) idea assistant. Enter your app idea and the problem it solves, and get instant suggestions for:
- MVP features
- Technical stack
- UI/UX design flow

The app uses a Mesop frontend and a LangGraph/OpenAI backend to generate concise, actionable product plans for startups and innovators.

## Features
- Simple web UI with two input fields: Idea & Problem
- Generates and displays:
  - MVP Features
  - Technical Recommendations
  - UI/UX Design Flow
- Powered by OpenAI and LangGraph

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install mesop
   ```

2. **Set your OpenAI API Key:**
   - Edit `backend.py` and set your API key in the `os.environ["OPENAI_API_KEY"]` line.

3. **Run the app:**
   ```bash
   mesop main.py
   ```
   Then open the provided URL (e.g. http://localhost:32123) in your browser.

## Project Structure
- `main.py` — Mesop frontend UI
- `backend.py` — LangGraph + OpenAI backend logic
- `requirements.txt` — Python dependencies

## License
Apache 2.0

---
Made with [Mesop](https://github.com/mesop-dev/mesop) and [LangGraph](https://github.com/langchain-ai/langgraph).
