import streamlit as st
import requests
import json
import os
import glob
from bs4 import BeautifulSoup
from pypdf import PdfReader
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# --- SETUP ---
load_dotenv() # Loads API keys from .env file

API_KEY = os.getenv("API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
REGION_URL = os.getenv("REGION_URL")
PDF_FILE = "matrix.pdf" # Ensure this file exists in the same folder!

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com", # Your resource list said 'us-south' (Dallas)
    "apikey": API_KEY
}

# --- 1. OFFICIAL KNOWLEDGE BASE (MULTI-PDF READER) ---
@st.cache_resource
def get_official_knowledge_base():
    """Reads ALL PDF files in the 'knowledge_base' folder."""
    combined_text = ""
    pdf_folder = "knowledge_base"
    
    if not os.path.exists(pdf_folder):
        print(f"⚠️ Folder '{pdf_folder}' not found.")
        return None
    
    # Find all PDF files in the folder
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    
    if not pdf_files:
        print("⚠️ No PDFs found in knowledge_base folder.")
        return None
        
    print(f"📄 Loading {len(pdf_files)} documents from Knowledge Base...")
    
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            # Read first 5 pages of EACH file (Safety limit to prevent crashing the AI)
            for i, page in enumerate(reader.pages[:5]): 
                combined_text += f"\n--- SOURCE: {os.path.basename(pdf_file)} (Page {i+1}) ---\n"
                combined_text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")
            
    return combined_text

# --- 2. STREET BRAIN (REDDIT SCRAPER) ---
def find_reddit_link(origin, destination):
    print(f"🔎 Searching Reddit for: {origin} to {destination}...")
    
    # query: searching within the subreddit for the route
    query = f'subreddit:HowToGetTherePH title:"{origin} to {destination}"'
    search_url = "https://www.reddit.com/search.json"
    
    # Get top 10 results (so we have backups if the first one is empty)
    params = {
        "q": query,
        "sort": "relevance", 
        "limit": 10 
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        r = requests.get(search_url, params=params, headers=headers)
        data = r.json()
        results = data.get('data', {}).get('children', [])
        
        if not results:
            print("❌ No results found.")
            return None

        # --- THE FIX: LOOP & FILTER ---
        print(f"   (Found {len(results)} potential threads. Checking for comments...)")
        
        for item in results:
            thread = item['data']
            title = thread.get('title', 'No Title')
            num_comments = thread.get('num_comments', 0)
            url = thread.get('url', '')
            
            # The Quality Check: Must have at least 2 comments to be useful
            if num_comments >= 2:
                print(f"✅ Picked Thread: '{title}' ({num_comments} comments)")
                return url
            else:
                print(f"   Skipping '{title}' (Only {num_comments} comments)...")
                
        print("❌ All found threads were empty/useless.")
        return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def get_reddit_text(url):
    # 1. Force the URL to use "old.reddit.com"
    # This interface is lighter and less protected against scrapers
    if "www.reddit" in url:
        scrape_url = url.replace("www.reddit", "old.reddit")
    elif "reddit.com" in url and "old.reddit" not in url:
        scrape_url = url.replace("reddit.com", "old.reddit.com")
    else:
        scrape_url = url
        
    # Remove query parameters to keep it clean
    scrape_url = scrape_url.split("?")[0]
    
    print(f"   ⬇️ Scraping OLD Reddit HTML: {scrape_url}...")

    # 2. Use a "Generic Linux" Header (Less suspicious for server-side requests)
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }
    
    try:
        r = requests.get(scrape_url, headers=headers, timeout=10)
        
        if r.status_code != 200:
            print(f"   ❌ Blocked: Status {r.status_code}")
            return None

        # 3. Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # 4. Extract Comments using Old Reddit's CSS Classes
        # The comment text is always inside <div class="usertext-body"> -> <div class="md">
        
        comments_found = []
        
        # Select all comment entries
        entries = soup.select('div.entry')
        
        for entry in entries:
            # Skip the main post (we just want comments) if needed, 
            # but usually the first entry is the post, subsequent are comments.
            
            # Find the author to skip AutoMod
            author_tag = entry.select_one('a.author')
            if author_tag:
                author = author_tag.text.strip()
                if author == "AutoModerator":
                    continue
            
            # Find the text body
            body_div = entry.select_one('div.usertext-body div.md')
            if body_div:
                text = body_div.get_text("\n").strip()
                
                # Filter garbage
                if text and text != "[deleted]" and "Welcome to r/HowToGetTherePH" not in text:
                    comments_found.append(text)
        
        if not comments_found:
            print("   ⚠️ No readable comments found.")
            return None

        print(f"   ✅ Successfully scraped {len(comments_found)} comments.")
        
        # Return the top 3 comments combined
        return "\n---\n".join(comments_found[:3])

    except Exception as e:
        print(f"   ❌ HTML Scrape Error: {e}")
        return None


# --- 3. THE TRUTH ENGINE (WATSONX RAG AGENT) ---
def analyze_commute(reddit_text, pdf_context, origin, dest):
    print("🧠 Watsonx is analyzing...")
    
    # Context Stuffing: Feed PDF data into the prompt
    pdf_section = f"OFFICIAL LTFRB FARE MATRIX DATA:\n{pdf_context[:5000]}" if pdf_context else "OFFICIAL DATA: Not Available."
    
    model = ModelInference(
        model_id="ibm/granite-3-8b-instruct", 
        params={
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MAX_NEW_TOKENS: 600
        },
        credentials=credentials,
        project_id=PROJECT_ID
    )
    
    prompt = f"""<|system|>
You are a Commute Truth Engine. 
Task 1: Extract commute steps from 'Street Advice'.
Task 2: Check 'Official Matrix' to see if the street fare is fair/legal.
Output JSON ONLY.

<|user|>
Origin: {origin}
Destination: {dest}

--- SOURCE 1: STREET ADVICE (Reddit) ---
"{reddit_text[:2000]}"

--- SOURCE 2: OFFICIAL LAWS (PDF Knowledge Base) ---
"{pdf_section}"

RESPONSE FORMAT:
[
  {{
    "step": 1,
    "mode": "Jeep/Bus",
    "details": "Route details here",
    "street_fare": 25 (integer or null),
    "official_check": "Compare street_fare vs PDF. If street fare is higher than official, warn 'Overprice'. If PDF missing, say 'No data'."
  }}
]

<|assistant|>
```json
"""
    response = model.generate_text(prompt=prompt)
    
    # Clean JSON
    clean_json = response.strip()
    if "```" in clean_json: clean_json = clean_json.replace("```json", "").replace("```", "")
    if "]" in clean_json: clean_json = clean_json[:clean_json.rfind("]")+1]
    
    try: return json.loads(clean_json)
    except: return []
# --- 4. MAIN UI ---
def main():
    st.set_page_config(page_title="CommutePH Truth Engine", page_icon="🇵🇭", layout="wide")
    st.title("🇵🇭 Commute Truth Engine")
    st.caption("Powered by **IBM Watsonx Granite** • **Reddit** • **LTFRB RAG**")
    
    # Load Knowledge Base
    pdf_text = get_official_knowledge_base()
    if pdf_text:
        st.success(f"✅ Active Knowledge Base: {PDF_FILE} (Official Data Loaded)")
    else:
        st.warning(f"⚠️ {PDF_FILE} not found. Running in 'Street Only' mode.")

    c1, c2 = st.columns(2)
    origin = c1.text_input("Origin", "Bacoor")
    dest = c2.text_input("Destination", "MOA")
    
    if st.button("Analyze Route", type="primary"):
        with st.status("🚀 Agents deploying...", expanded=True):
            
            st.write("🔎 Agent 1: Searching community threads...")
            link = find_reddit_link(origin, dest)
            
            if link:
                st.write(f"⬇️ Agent 2: Scraping {link}...")
                reddit_text = get_reddit_text(link)
                
                if reddit_text:
                    st.write("⚖️ Agent 3: Cross-referencing Street vs. Official Data...")
                    routes = analyze_commute(reddit_text, pdf_text, origin, dest)
                    
                    st.divider()
                    for r in routes:
                        with st.container():
                            st.subheader(f"Step {r.get('step')}: {r.get('mode')}")
                            st.write(r.get('details'))
                            
                            fare = r.get('street_fare')
                            check = r.get('official_check')
                            
                            if fare:
                                col_a, col_b = st.columns(2)
                                col_a.metric("Street Price", f"₱{fare}")
                                
                                # Verification Logic
                                if "overprice" in str(check).lower() or "higher" in str(check).lower():
                                    col_b.error(f"⚠️ {check}")
                                else:
                                    col_b.success(f"✅ {check}")
                            else:
                                st.info(f"ℹ️ {check}")
                else:
                    st.error("Could not read Reddit comments.")
            else:
                st.error("No route found.")

if __name__ == "__main__":
    main()