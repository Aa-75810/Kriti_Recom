import streamlit as st
from streamlit.web import cli as stcli
from fastapi import FastAPI , Request
import threading
import jsonify
import uvicorn
from pydantic import BaseModel
from Google import create_service
import pandas as pd
import joblib
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, VideoUnavailable
import re
from transformers import MBartTokenizer, MBartForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Constants
CLIENT_SECRETS_FILE = "D:\\infominez\\Kriti_recom\\recomm\\token files\\client_secret.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]
API_KEY = "AIzaSyDc0rLG5MxomNV9jZWRFv0B5ke3vXiAIzw"
CHANNEL_ID = "UCe2JAC5FUfbxLCfAvBWmNJA"
CSV_PATH = "D:\\infominez\\Kriti_recom\\recomm\\try.csv"
MODEL_PATH = "recipe_recommendation_model.pkl"
app = FastAPI()

app.state.credentials = ""

def authenticate_user():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
    credentials = flow.run_local_server(port=8080, access_type="offline")
    return credentials

def fetch_transcript(video_id):
    lang = ['en', 'hi', "hi-IN"]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=lang)
        return " ".join([entry['text'] for entry in transcript])
    except (TranscriptsDisabled, VideoUnavailable):
        return "Transcript not available."
    except Exception as e:
        return f"Error: {e}"

def parse_ingredients_and_methods(description):
    ingredients_match = re.search(r"(?i)(?:Ingredients|INGREDIENTS):(.+?)(?:Method|METHOD|$)", description, re.DOTALL)
    ingredients = [line.strip() for line in ingredients_match.group(1).split("\n") if line.strip()] if ingredients_match else []

    methods_match = re.search(r"(?i)(?:Method|METHOD):(.+)", description, re.DOTALL)
    methods = [line.strip() for line in methods_match.group(1).split("\n") if line.strip()] if methods_match else []

    return ingredients, methods

def get_uploaded_videos(credentials, query="paneer makhani"):
    youtube = build("youtube", "v3", credentials=credentials)
    response = youtube.search().list(part="snippet", q=query, type="video", maxResults=50).execute()
    videos = []
    for item in response.get("items", []):
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        if "#shorts" in title.lower():
            continue
        description = item["snippet"]["description"]
        ingredients, methods = parse_ingredients_and_methods(description)
        transcript = fetch_transcript(video_id) if not ingredients or not methods else "Not required"
        videos.append({
            "video_id": video_id,
            "title": title,
            "description": description,
            "ingredients": "\n".join(ingredients) if ingredients else "Not available",
            "methods": "\n".join(methods) if methods else "Not available",
            "transcript": transcript
        })
    return videos

def save_video_dataframe(videos):
    try:
        database = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        database = pd.DataFrame()
    new_df = pd.DataFrame(videos)
    updated_df = pd.concat([database, new_df]).drop_duplicates(subset="video_id").reset_index(drop=True)
    updated_df.to_csv(CSV_PATH, index=False)
    return updated_df



# Tokenize and prepare data with mBART
def tokenize_csv(CSV_PATH, tokenizer_path):
    df = pd.read_csv(CSV_PATH)
    tokenizer = MBartTokenizer.from_pretrained(tokenizer_path)
    df = df.fillna("Not available")
    df['Tokenized'] = df['title'] + ' ' + df['description']
    df['Tokenized'] = df['Tokenized'].apply(lambda x: tokenizer.encode(x, return_tensors='pt'))
    return df

# Train or update the recommendation model
def train_model(CSV_PATH, model_path):
    df = pd.read_csv(CSV_PATH)
    df = df.fillna("Not available")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['title'] + ' ' + df['description'])
    
    # Save model and vectorizer
    joblib.dump((tfidf_matrix, vectorizer, df), model_path)

def recommend_recipe(user_input, model_path):
    tfidf_matrix, vectorizer, df = joblib.load(model_path)
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, tfidf_matrix)
    top_indices = similarities.argsort()[0][-10:][::-1]
    return df.iloc[top_indices]

# Streamlit UI
API_URL = "http://127.0.0.1:8000/recommend"
st.title("Kriti Oil Recipe Recommender")

if st.button("Authenticate YouTube API"):
    credentials = authenticate_user()
    st.session_state['credentials'] = credentials
    st.success("Authenticated successfully!")

user_input = st.text_input("Enter an ingredient or recipe name to get recommendations:")
# if st.button("Fetch YouTube Data"):
df = pd.read_csv(CSV_PATH)
tokenizer_path = "facebook/mbart-large-cc25"
if st.button("Get Recommendations"):
    if 'credentials' in st.session_state:
        if user_input in df["title"]:
            st.success("Fetching data...")
            recommend_recipe= recommend_recipe(user_input, MODEL_PATH)
        else:
            videos = get_uploaded_videos(st.session_state['credentials'], user_input)
            df = save_video_dataframe(videos)
            tokenize_csv(CSV_PATH, tokenizer_path=tokenizer_path)
            train_model(CSV_PATH , MODEL_PATH)
            recommendations = recommend_recipe(user_input, MODEL_PATH)
            # st.write(df)
    else:
        st.error("Please authenticate first.")
    # recommendations = recommend_recipe(user_input, MODEL_PATH)
    # st.write(recommendations)

# if st.button("Show Saved Recipes"):
#     try:
#         saved_df = pd.read_csv(CSV_PATH)
#         st.write(saved_df)
#     except FileNotFoundError:
#         st.error("No saved data found.")

st.text("Developed for Kriti Oil Company")

# retrieve API


class RecipeRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    # if not authenticate_user():
    #     return {"error" : "Authentication Failed"}
    credentials = authenticate_user()
    app.state.credentials = credentials
    return {"message": "Authentication successful"}
    # return {"message" : "Fastapi is running! use post /authenticate"}

# def set_credentials(credentials):
#     app.state.credentials = credentials

# @app.get("/authenticate")
# def authenticate():
#     credentials = authenticate_user()
#     app.state.credentials = credentials
#     return {"message": "Authentication successful"}

@app.post("/recommend")
def recommend_recipe_api(request:RecipeRequest) :
    user_input = request.query
    recommendations = []
    df = pd.read_csv(CSV_PATH)

    # app.state.credentials = authenticate_user()
    # # print("appstream::::::",app.state)
    # # print("app.state.credentials:", getattr(app.state, 'credentials', None))
    # # creds = getattr(app.state, 'credentials', None)
    # # # if creds and app.state.credentials:
    if getattr(app.state, 'credentials', None):
        print("hello crdekjlsdjflk::",app.state.credentials)
        print("hello crdekjlsdjflk::",df["title"])
        print("paneer tikka:::::::::::::::::::::::::::::",df[df["title"].str.contains(user_input, case=False, na=False)])
        matching_rows = df[df["title"].str.contains(user_input, case=False, na=False)]
        if not matching_rows.empty:
            print("Fetching data...")
            recommendations= recommend_recipe(user_input, MODEL_PATH)
        else:
            videos = get_uploaded_videos(app.state.credentials, user_input)
            df = save_video_dataframe(videos)
            tokenize_csv(CSV_PATH, tokenizer_path=tokenizer_path)
            train_model(CSV_PATH , MODEL_PATH)
            recommendations = recommend_recipe(user_input, MODEL_PATH)
                # st.write(df)
    else:
        return {"error": "Please authenticate first."}
    print("recommendations",recommendations)   
    return recommendations



# import subprocess
def run_fastapi():
    uvicorn.run(app , port=8000)
    # subprocess.Popen(["uvicorn" , "api_try1:app" , "--host", "127.0.0.1", "--port", "8000", "--reload"])

# run_fastapi()
# Run Fastapi in separate thread
threading.Thread(target=run_fastapi , daemon=True).start()