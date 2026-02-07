import streamlit as st
import requests

# Point this to your local API (or your Ngrok URL if testing that)
API_URL = "http://localhost:8000/analyze_route"

st.set_page_config(page_title="Commute Truth Engine", page_icon="🇵🇭", layout="centered")

# Header Section
st.title("🇵🇭 Commute Truth Engine")
st.markdown("""
**Architecture:** `Reddit Scraper`` ➝ `LTFRB PDF RAG` ➝ `IBM Watsonx Granite` ➝ FastAPI`
""")

# Input Section
with st.container(border=True):
    c1, c2 = st.columns(2)
    origin = c1.text_input("Origin", "Bacoor")
    dest = c2.text_input("Destination", "MOA")
    
    analyze_btn = st.button("Analyze Route", type="primary", use_container_width=True)

# Results Section
if analyze_btn:
    with st.status("Agents are deploying...", expanded=True) as status:
        st.write("Searching r/HowToGetTherePH...")
        
        try:
            payload = {"origin": origin, "destination": dest}
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("status") == "success":
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                    
                    # Show Source
                    st.success(f"**Knowledge Source:** [View Reddit Thread]({result.get('source')})")
                    
                    # Show Steps
                    routes = result.get("data", [])
                    for r in routes:
                        with st.container(border=True):
                            # Header
                            c_step, c_mode = st.columns([1, 4])
                            c_step.metric("Step", r.get('step'))
                            c_mode.subheader(f"{r.get('mode')}")
                            
                            # Details
                            st.info(r.get('details'))
                            
                            # Official Check
                            check = r.get('official_check')
                            if check and "verify" in str(check).lower():
                                st.caption(f"🏛️ **Official Data:** {check}")
                            else:
                                st.caption(f"🏛️ **Official Data:** {check}")

                else:
                    status.update(label="API Error", state="error")
                    st.error(result.get("message"))
            else:
                status.update(label="Server Connection Error", state="error")
                st.error(f"Status Code: {response.status_code}")
                
        except Exception as e:
            status.update(label="Connection Failed", state="error")
            st.error(f"Is api.py running? {e}")