import os
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.utilities import GoogleSerperAPIWrapper
from duckduckgo_search import DDGS
from transformers import pipeline


llm = OllamaLLM(model="tinyllama:chat")
prompt = PromptTemplate( input_variables=["query"], template="User: {query}\nAssistant:")
llm_chain = LLMChain(llm=llm, prompt=prompt)

search_tool = GoogleSerperAPIWrapper()

summarizer = pipeline("summarization", model="t5-small", device=-1)

app = FastAPI(title="Hybrid Chatbot")

def needs_search(llm_text: str, user_query: str) -> bool:
    """Basic rules to decide if we should fetch live web data."""
    for kw in ("recent", "latest", "current", "today", "news"):
        if kw in user_query.lower():
            return True
    if len(llm_text) < 50:
        return True
    for phrase in ("i don't know", "as a language model"):
        if phrase in llm_text.lower():
            return True
    return False


@app.post("/chat/llm_only")
async def chat_llm_only(req: Request):
    data = await req.json()
    q = data.get("prompt")
    if not q:
        raise HTTPException(400, "Missing ‘prompt’")
    resp = llm_chain.predict(query=q)
    return {"response": resp}


@app.get("/search")
async def search_only(q: str):
    snippets = []
    
    try:
        docs = search_tool.results(q).get("organic", [])
        snippets = [d["snippet"] for d in docs[:5]]
    except Exception as e:
        print(f"Serper API error: {str(e)}")
    
    if not snippets:
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(q, max_results=5)]
                snippets = [r["body"] for r in results]
        except Exception as e:
            print(f"DuckDuckGo error: {str(e)}")
            return {"error": "Search services unavailable", "query": q}

    if not snippets:
        return {"error": "No results found", "query": q}

    try:
        combined = " ".join(snippets)
        summary = summarizer(combined, max_length=100, min_length=30)[0]["summary_text"]
        return {"query": q, "summary": summary}
    except Exception as e:
        print(f"Summarization error: {str(e)}")
        return {"query": q, "summary": snippets[0] if snippets else "No summary available"}


@app.post("/chat/hybrid")
async def chat_hybrid(req: Request):
    data = await req.json()
    q = data.get("prompt")
    if not q:
        raise HTTPException(400, "Missing ‘prompt’")

    llm_resp = llm_chain.predict(query=q)

    if needs_search(llm_resp, q):
        docs = search_tool.results(q).get("organic", [])
        snippets = [d["snippet"] for d in docs[:5]]
        if not snippets:
            snippets = [r["body"] for r in DDGS().text(q, max_results=5)]

        combined = " ".join(snippets)
        summary = summarizer(combined, max_length=100, min_length=30)[0]["summary_text"]

        hybrid_prompt = (
            f"[WEB SUMMARY]\n{summary}\n\n"
            f"[ORIGINAL QUESTION]\n{q}"
        )
        llm_resp = llm_chain.predict(query=hybrid_prompt)

    return {"response": llm_resp}