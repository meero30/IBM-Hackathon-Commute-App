# Commute Truth Engine

FastAPI backend that generates **step-by-step commute instructions**
between two locations in the Philippines.

The API collects real commuter advice from Reddit, verifies it against
official transport documents (PDFs), and synthesizes a route using an
IBM Watsonx (Granite) LLM.

This service is designed to be connected to **Watsonx Orchestrate** as a
tool for an AI agent.



## What It Does

1.  Searches Reddit (r/HowToGetTherePH) for commute discussions.
2.  Filters results to match the correct travel direction (Origin →
    Destination).
3.  Scrapes the thread title, post, and top comments.
4.  Cross-checks information with official transport PDF files.
5.  Uses IBM Granite model to generate structured commute steps.
6.  Returns a clean JSON route guide.

NOTE: I've also added a streamlit code file for trying out the API outside watsonx orchestrate.


## Tech Stack

-   FastAPI
-   IBM Watsonx AI (Granite 3 8B Instruct)
-   Reddit search + scraping (BeautifulSoup)
-   PDF parsing (pypdf)
-   Ngrok (public endpoint)
-   Watsonx Orchestrate (agent integration)



## Project Structure

    project/
    │── api.py
    │── knowledge_base/
    │    ├── routes_1_example.pdf
    │    ├── fares_1_example.pdf
    │── .env
    │── requirements.txt

`knowledge_base/` contains official documents used to verify fares and
routes.


## Requirements

Python 3.10+

Install dependencies:

``` bash
pip install fastapi uvicorn requests beautifulsoup4 python-dotenv pypdf pyngrok ibm-watsonx-ai
```

or use the requirements.txt file

``` bash
pip install -r requirements.txt
```



## Environment Variables

Create a `.env` file:

    API_KEY=your_watsonx_api_key
    PROJECT_ID=your_project_id
    REGION_URL=https://us-south.ml.cloud.ibm.com
    NGROK_AUTH_TOKEN=your_ngrok_token



## Running the Server

``` bash
python api.py
```

On startup the API will:

-   Load PDFs from `knowledge_base/`
-   Open a public URL via Ngrok
-   Print the public API endpoint

Example:

    PUBLIC API URL: https://xxxx.ngrok-free.app



## API Endpoint

### Analyze Route

**POST** `/analyze_route`

Request:

``` json
{
  "origin": "Bacoor",
  "destination": "MOA"
}
```

Response:

``` json
{
  "status": "success",
  "data": [
    {
      "step": 1,
      "mode": "Jeep/Bus",
      "details": "Specific instruction.",
      "official_check": "Verification result."
    }
  ],
  "source": "https://reddit.com/..."
}
```


## Knowledge Base (Official Data)

Place official transport PDFs inside:

    knowledge_base/

The API automatically reads the first 10 pages of each PDF and feeds it
to the LLM as verification data. Although none of the PDFS I pasted were 10 pages long.
All of the PDFs were acquired in the LTFRB website: https://ltfrb.gov.ph/



## How Direction Filtering Works

The system prevents incorrect advice by:

-   Searching Reddit for `Origin to Destination` in titles
-   Ignoring reverse routes
-   Using thread titles as validation signals
-   Only allowing forward-direction commute steps



## Watsonx Orchestrate Integration

Use the Ngrok public URL as a **custom tool API** inside Watsonx
Orchestrate.

The agent can call:

    POST /analyze_route

and receive structured commute instructions it can directly present to
the user.



## Notes

-   The API does not invent routes; it synthesizes from real commuter
    advice.
-   Acronyms (e.g., PITX, LRT, UV, FX) are preserved intentionally.
-   Accuracy improves as more official PDFs are added.


