import os
import glob
import json
import requests
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from bs4 import BeautifulSoup
from pypdf import PdfReader
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from pyngrok import ngrok

# --- 1. CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
REGION_URL = os.getenv("REGION_URL")
NGROK_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
PDF_FOLDER = "knowledge_base"
KNOWLEDGE_BASE_TEXT = ""

# --- 2. LIFESPAN (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global KNOWLEDGE_BASE_TEXT
    print("API (Direction-Aware) Starting...")

    # Load PDFs
    if os.path.exists(PDF_FOLDER):
        pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
        text_data = ""
        for pdf_file in pdf_files:
            try:
                reader = PdfReader(pdf_file)
                for i, page in enumerate(reader.pages[:10]):
                    text_data += f"\n--- SOURCE: {os.path.basename(pdf_file)} ---\n" + page.extract_text() + "\n"
            except: pass
        KNOWLEDGE_BASE_TEXT = text_data
        print(f"Loaded Knowledge Base ({len(KNOWLEDGE_BASE_TEXT)} chars)")

    # Start Ngrok
    if NGROK_TOKEN:
        ngrok.set_auth_token(NGROK_TOKEN)
        try:
            public_url = ngrok.connect(8000).public_url
            print(f"\nPUBLIC API URL: {public_url}\n")
        except Exception as e: print(f"Ngrok Error: {e}")
    
    yield
    ngrok.kill()

app = FastAPI(title="Commute Truth Engine V4", lifespan=lifespan)

class RouteRequest(BaseModel):
    origin: str
    destination: str

# --- 3. SMARTER SEARCH LOGIC ---
def search_reddit_threads(origin, destination):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    base_url = "https://www.reddit.com/search.json"
    
    # ATTEMPT 1: Strict Direction Search (e.g. title:"Bacoor to MOA")
    # This filters out the reverse routes immediately.
    strict_query = f'subreddit:HowToGetTherePH title:"{origin} to {destination}"'
    print(f"Attempt 1 (Strict): {strict_query}")
    
    try:
        r = requests.get(base_url, params={"q": strict_query, "sort": "top", "limit": 3}, headers=headers)
        results = r.json().get('data', {}).get('children', [])
        
        valid_urls = []
        for item in results:
            if item['data'].get('num_comments', 0) > 0:
                valid_urls.append(item['data']['url'])
        
        # If strict search works, return ONLY these (High Confidence)
        if valid_urls:
            print(f"Found {len(valid_urls)} strict matches.")
            return valid_urls
            
    except: pass

    # ATTEMPT 2: Fallback to Keywords (If strict failed)
    print("Strict search failed. Falling back to keywords...")
    loose_query = f'subreddit:HowToGetTherePH "{origin}" "{destination}"'
    try:
        r = requests.get(base_url, params={"q": loose_query, "sort": "relevance", "limit": 5}, headers=headers)
        results = r.json().get('data', {}).get('children', [])
        valid_urls = []
        for item in results:
            if item['data'].get('num_comments', 0) > 1:
                valid_urls.append(item['data']['url'])
        return valid_urls[:3]
    except: return []

def scrape_thread_content(url):
    if "www.reddit" in url: url = url.replace("www.reddit", "old.reddit")
    url = url.split("?")[0]
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, 'html.parser')
        content = []
        # Get Title to help AI check direction
        title = soup.select_one('a.title')
        if title: content.append(f"THREAD TITLE: {title.get_text().strip()}")
        
        main_post = soup.select_one('div.expando div.md')
        if main_post: content.append(f"OP POST: {main_post.get_text().strip()}")
        
        for entry in soup.select('div.entry')[:4]:
            body = entry.select_one('div.usertext-body div.md')
            if body:
                text = body.get_text("\n").strip()
                if text and "Welcome to" not in text: content.append(f"COMMENT: {text}")
        return "\n".join(content)
    except: return ""

# --- 4. DIRECTION-AWARE ENDPOINT ---
@app.post("/analyze_route")
async def analyze_route_endpoint(request: RouteRequest):
    threads = search_reddit_threads(request.origin, request.destination)
    if not threads:
        return {"status": "error", "message": "No relevant threads found."}
    
    street_data = ""
    for i, url in enumerate(threads):
        street_data += f"\n--- THREAD {i+1} ({url}) ---\n" + scrape_thread_content(url)
    
    print("Watsonx is synthesizing (with Direction Check)...")
    
    prompt = f"""<|system|>
You are an expert Commute Guide. 
The user wants to go FROM {request.origin} TO {request.destination}.

CRITICAL INSTRUCTIONS:
1. Check the "THREAD TITLE" in the street data. 
2. If a thread is about the REVERSE route ({request.destination} to {request.origin}), IGNORE IT unless it explicitly mentions how to go back.
3. Only provide steps that move FROM {request.origin} TO {request.destination}.
4. If valid steps are found, verify fares against OFFICIAL DATA.
5. DO NOT Invent meanings of Acronyms. Just leave the acronyms as-is.


<|user|>
Request: {request.origin} to {request.destination}

STREET ADVICE:
{street_data[:5000]}

OFFICIAL DATA:
{KNOWLEDGE_BASE_TEXT[:3000]}

RESPONSE FORMAT (JSON ONLY):
[
  {{
    "step": 1,
    "mode": "Jeep/Bus",
    "details": "Specific instruction.",
    "official_check": "Verification result."
  }}
]
<|assistant|>
```json
"""
    try:
        model = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            params={GenParams.DECODING_METHOD: "greedy", GenParams.MAX_NEW_TOKENS: 900},
            credentials={"url": REGION_URL, "apikey": API_KEY},
            project_id=PROJECT_ID
        )
        res = model.generate_text(prompt=prompt)
        
        # Cleanup
        clean = res.strip().replace("```json", "").replace("```", "")
        start_idx = clean.find("[")
        if start_idx != -1: clean = clean[start_idx:]
        
        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            if "Extra data" in str(e):
                clean = clean[:e.pos]
                data = json.loads(clean)
            elif "]" in clean:
                clean = clean[:clean.rfind("]")+1]
                data = json.loads(clean)
            else: raise e

        return {"status": "success", "data": data, "source": threads[0]}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)