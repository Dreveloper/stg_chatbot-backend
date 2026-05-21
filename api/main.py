from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="STG Clinical RAG API")

rag_chain = None

def get_chain():
    global rag_chain
    if rag_chain is None:
        from rag.chain import get_rag_chain
        rag_chain = get_rag_chain()
    return rag_chain

class QuestionRequest(BaseModel):
    question: str

@app.on_event("startup")
def startup_event():
    if not os.getenv("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY not set. API will return errors.")
    if not os.getenv("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY not set. API will return errors.")

@app.get("/", response_class=HTMLResponse)
def chat_ui():
    return HTMLResponse("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>STG Clinical Assistant</title>
<script src="https://cdn.jsdelivr.net/npm/marked@15.0.7/marked.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f5f5;
    color: #1a1a1a;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    background: #fff;
    border-bottom: 1px solid #e0e0e0;
    padding: 16px 24px;
    text-align: center;
  }
  header h1 { font-size: 18px; font-weight: 600; color: #1a1a1a; }
  header p { font-size: 13px; color: #666; margin-top: 2px; }
  #chat {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .msg {
    max-width: 720px;
    width: fit-content;
    padding: 12px 16px;
    border-radius: 12px;
    font-size: 14px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .msg.user {
    background: #e8e8e8;
    align-self: flex-end;
    border-bottom-right-radius: 4px;
  }
  .msg.bot {
    background: #fff;
    border: 1px solid #e0e0e0;
    align-self: flex-start;
    border-bottom-left-radius: 4px;
  }
  .msg.bot strong { font-weight: 600; }
  .msg.bot ul, .msg.bot ol { margin: 6px 0; padding-left: 20px; }
  .msg.bot li { margin: 2px 0; }
  .msg.bot hr { border: none; border-top: 1px solid #e0e0e0; margin: 12px 0; }
  .msg.bot em { font-style: italic; }
  .msg.bot p { margin: 6px 0; }
  .msg.bot.loading { color: #999; }
  #input-area {
    background: #fff;
    border-top: 1px solid #e0e0e0;
    padding: 16px 24px;
    display: flex;
    gap: 8px;
  }
  #input-area input {
    flex: 1;
    padding: 10px 14px;
    border: 1px solid #d0d0d0;
    border-radius: 8px;
    font-size: 14px;
    outline: none;
    background: #fafafa;
  }
  #input-area input:focus { border-color: #888; background: #fff; }
  #input-area button {
    padding: 10px 20px;
    background: #1a1a1a;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    cursor: pointer;
    font-weight: 500;
  }
  #input-area button:hover { background: #333; }
  #input-area button:disabled { background: #999; cursor: not-allowed; }
</style>
</head>
<body>
<header>
  <h1>STG Clinical Assistant</h1>
  <p>Tanzania Standard Treatment Guidelines</p>
</header>
<div id="chat"></div>
<div id="input-area">
  <input id="input" type="text" placeholder="Ask a question..." autofocus>
  <button id="send" onclick="ask()">Send</button>
</div>
<script>
  const chat = document.getElementById('chat');
  const input = document.getElementById('input');
  const sendBtn = document.getElementById('send');

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function addMsg(text, role) {
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    if (role === 'bot') {
      div.innerHTML = marked.parse(text);
    } else {
      div.textContent = text;
    }
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div;
  }

  async function ask() {
    const q = input.value.trim();
    if (!q) return;
    input.value = '';
    sendBtn.disabled = true;
    addMsg(q, 'user');
    const loading = addMsg('Thinking...', 'bot loading');
    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q })
      });
      const data = await res.json();
      loading.remove();
      if (data.answer) addMsg(data.answer, 'bot');
      else addMsg('Error: ' + (data.detail || 'Unknown error'), 'bot');
    } catch (e) {
      loading.remove();
      addMsg('Error: ' + e.message, 'bot');
    }
    sendBtn.disabled = false;
    input.focus();
  }

  input.addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });
</script>
</body>
</html>""")

@app.get("/health")
def health_check():
    status = "healthy" if os.getenv("GROQ_API_KEY") and os.getenv("GOOGLE_API_KEY") else "degraded"
    return {"status": status}

@app.post("/ask")
def ask_question(payload: QuestionRequest):
    try:
        chain = get_chain()
        answer = chain.invoke(payload.question)
        return {"answer": answer}
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")