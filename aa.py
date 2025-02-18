import os
import pandas as pd
import Google
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from transformers import MBartTokenizer, MBartForConditionalGeneration
from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled, VideoUnavailable
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import joblib

# Step 1: Path to your 'client_secret.json' file
CLIENT_SECRETS_FILE = "D:\\infominez\\Kriti_recom\\recomm\\token files\\client_secret.json"

# Step 2: Define the required scope for YouTube API
SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]

def authenticate_user():
    # Step 3: Create OAuth 2.0 flow using the client secrets file
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
    
    # Step 4: Run the OAuth flow and open the local server for user authentication
    credentials = flow.run_local_server(port=8080, access_type="offline")

    print("Access_Token : " , credentials.token)

    print("Refresh_Token : " , credentials.refresh_token)

    # Step 5: Return the credentials (contains access token and refresh token)
    return credentials

def get_video_details(credentials, video_id):
    # Step 6: Build the YouTube Data API client
    youtube = build("youtube", "v3", credentials=credentials)

    print("youtube:", youtube)

    # Step 7: Request the video details
    request = youtube.videos().list(
        part="snippet",  # Use a properly formatted string
        id=video_id  # Video ID for which details are requested
    )
    response = request.execute()

    # Step 8: Return the video details
    return response

channel_id = "UCe2JAC5FUfbxLCfAvBWmNJA"
def validate_channel_ownership(channel_info):
    for channel in channel_info.get("items", []):
        if channel["id"] == channel_id:
            print(f"User authenticated successfully for channel: {channel_id}")
            return True
    print("Authentication failed. User does not own the specified channel.")
    return False

def parse_ingredients_and_methods(description):
    """
    Extract ingredients and methods from the description, if present.
    """
    # ingredients = []
    # methods = []

    # if "Ingredients:" in description:
    #     ingredients_section = description.split("Ingredients:")[1].split("Method:")[0].strip()
    #     ingredients = ingredients_section.split("\n")
    # if "Method:" in description:
    #     methods_section = description.split("Method:")[1].strip()
    #     methods = methods_section.split("\n")

    # return ingredients, methods
    ingredients = []
    methods = []

    # Extract ingredients if "INGREDIENTS" section is present
    ingredients_match = re.search(r"(?i)(?:Ingredients|INGREDIENTS):(.+?)(?:Method|METHOD|$)", description, re.DOTALL)
    if ingredients_match:
        ingredients_section = ingredients_match.group(1).strip()
        ingredients = [line.strip() for line in ingredients_section.split("\n") if line.strip()]

    # Extract methods if "METHOD" section is present
    methods_match = re.search(r"(?i)(?:Method|METHOD):(.+)", description, re.DOTALL)
    if methods_match:
        methods_section = methods_match.group(1).strip()
        methods = [line.strip() for line in methods_section.split("\n") if line.strip()]

    return ingredients, methods

def fetch_transcript(video_id):
    """
    Fetch the transcript of the video using youtube_transcript_api.
    """
    lang = ['en', 'hi', "hi-IN"]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id , languages=lang)
        transcript_text = " ".join([entry['text'] for entry in transcript])
        return transcript_text
    except TranscriptsDisabled:
        return "Transcript not available (disabled for this video)."
    except VideoUnavailable:
        return "Transcript not available (video unavailable)."
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"
    
import time
import pandas as pd

def get_uploaded_videos(credentials, query = "paneer makhani"):
    """
    Retrieve all videos uploaded by the authenticated user's channel.
    """
    youtube = build("youtube", "v3", credentials=credentials)
    
    print("youtube:", youtube)

    all_video= []
    next_page_token = None
    # Get the channel ID of the authenticated user
    channels_response = youtube.search().list(
        part="snippet",
        q=query,
        # mine=True  # Get details for the authenticated user's channel
        type="video",
        maxResults = 20 #50
        # pageToken=next_page_token
    ).execute()
    print("channels_response : " , channels_response)

    channel_id = channels_response.get("items",[])[0]["id"]
    print("Channel ID: " , channel_id)

    # uploads_playlist_id = []
    # uploads_playlist_title = []
    # for channel in channels_response["items"]:
    #     # uploads_playlist_id = channel["id"]["videoId"]
    #     # uploads_playlist_title = channel["snippet"]["title"]
    #     # uploads_playlist_url = channel["thumbnails"]["default"]
    #     uploads_playlist_id.append(channel["id"]["videoId"])
    #     uploads_playlist_title.append(channel["snippet"]["title"])
    #     # print("upload playlist", uploads_playlist_id)
    #     # print("upload titles", uploads_playlist_title)
    #     # print("upload playlist", uploads_playlist_url)

    uploads_playlist = []  # Initialize as an empty list to store dictionaries

    for channel in channels_response["items"]:
        # Create a dictionary with videoId and title
        channel_info = {
            "videoId": channel["id"]["videoId"],
            "title": channel["snippet"]["title"]
        }
        uploads_playlist.append(channel_info)

    # Uncomment to view the result
    print("Uploads Playlist:", uploads_playlist)

    # if not uploads_playlist_id:
    #     print("Failed to find the uploads playlist ID for the channel.")
    #     return None

    # print(f"Uploads Playlist ID: {uploads_playlist_id}")
    # print(f"Uploads Playlist tile: {uploads_playlist_title}")
   

    # Fetch all videos in the uploads playlist
    videos = []
    next_page_token = None
    
    for channel in uploads_playlist:
        id = channel["videoId"]
        # print("id is ::: ", id)
        playlist_items_response = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=id
        ).execute()
        print("playlist_items_response", playlist_items_response)
        # Extract the description
        if "items" in playlist_items_response and len(playlist_items_response["items"]) > 0:
            video = playlist_items_response["items"][0]  # Get the first video
            description = video["snippet"]["description"]  # Extract description
            ingredients , methods = parse_ingredients_and_methods(description)

            print("Description:")
            print(description)

            transcript = None
            if not ingredients or not methods:
                transcript = fetch_transcript(id)
            # ingredients = video["snippet"]["ingredients"]
            # methods  = video["snippet"]["methods"]
            
            print("ingredients : ")
            print(ingredients)
            print("methods : ")
            print(methods)

            videos.append({"video_id": channel["videoId"], "title": channel["title"], "description": description,
                           "ingredients": "\n".join(ingredients) if ingredients else "Not available",
                            "methods": "\n".join(methods) if methods else "Not available",
                            "transcript": transcript if transcript else "Not required"})
        else:
            print("No videos found in the response.")
    
    print('Videos : ', videos)
    return videos

def display_videos(videos):
    """
    Display the retrieved video details.
    """
    for video in videos:
        print(f"Video ID: {video['video_id']}")
        print(f"Title: {video['title']}")
        print(f"Description: {video['description']}")
        # print(f"Ingredients:{video['ingredients']}")
        # print(f"Methods:{video['methods']}")
        print("-" * 50)

import os

def save_video_dataframe(videos ):
    database = pd.read_csv("try.csv")
    new_df = pd.DataFrame(videos)

    if database is not None:
        update_df = pd.concat([database,new_df]).drop_duplicates(subset="video_id").reset_index(drop=True)
        
    else:
        update_df=new_df

    print("video_dataframe : " , update_df)
    # update_df.to_csv("video_dataframe.csv" , index = False)
    update_df.to_csv("try.csv" , index = False)

    # if os.path.exists("try.csv"):
    #     user_input = input(f"The file '{update_df}' already exists. Do you want to update it? (yes/no): ").strip().lower()
    #     if user_input == "yes":
    #     # Save the updated DataFrame to the file
    #         update_df.to_csv("try.csv", index=False)
    #         print(f"The file '{update_df}' has been updated.")
    #     else:
    #         print("The file was not updated.")
    # else:
    #     # Save the file directly if it doesn't exist
    #     update_df.to_csv("try.csv", index=False)
    #     print(f"The file '{update_df}' has been created.")

    return update_df

# Set up YouTube Data API
def fetch_youtube_data(credentials,api_key, channel_id, csv_path, user_query="paneer makhani"):
    youtube = build('youtube', 'v3', developerKey=api_key , credentials=credentials)
    
    request = youtube.search().list(
        part='snippet',
        channelId=channel_id,
        q=user_query,
        type='video',
        maxResults=50
    )
    response = request.execute()
    # channel_id = response.get("items",[])[0]["id"]
    # print("Channel ID: " , channel_id)
    
    data = []
    for item in response['items']:
        video_title = item['snippet']['title']
        description = item['snippet']['description']
        video_id = item['id']['videoId']
        # return ingredients, methods
        ingredients = []
        methods = []

        # Extract ingredients if "INGREDIENTS" section is present
        ingredients_match = re.search(r"(?i)(?:Ingredients|INGREDIENTS):(.+?)(?:Method|METHOD|$)", description, re.DOTALL)
        if ingredients_match:
            ingredients_section = ingredients_match.group(1).strip()
            ingredients = [line.strip() for line in ingredients_section.split("\n") if line.strip()]

        # Extract methods if "METHOD" section is present
        methods_match = re.search(r"(?i)(?:Method|METHOD):(.+)", description, re.DOTALL)
        if methods_match:
            methods_section = methods_match.group(1).strip()
            methods = [line.strip() for line in methods_section.split("\n") if line.strip()]

        def fetch_transcript(video_id):
          """
          Fetch the transcript of the video using youtube_transcript_api.
          """
          lang = ['en', 'hi', "hi-IN"]
          try:
              transcript = YouTubeTranscriptApi.get_transcript(video_id , languages=lang)
              transcript_text = " ".join([entry['text'] for entry in transcript])
              return transcript_text
          except TranscriptsDisabled:
              return "Transcript not available (disabled for this video)."
          except VideoUnavailable:
              return "Transcript not available (video unavailable)."
          except Exception as e:
              return f"Error fetching transcript: {str(e)}"
          
        data.append({'Title': video_title, 'Description': description, 'Video_ID': video_id})
        if not data:
            print("No new data fetched from YouTube.")
        print(data)

        return ingredients, methods
    

    # channel_id = response.get("items",[])[0]["id"]
    # print("Channel ID: " , channel_id)

    # Update CSV
    df = pd.DataFrame(data)
    
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        df = pd.concat([existing_df, df]).drop_duplicates(subset=['Video_ID']).reset_index(drop=True)
        
    df.to_csv(csv_path, index=False)

    return df

# credentials = authenticate_user()
# api_key = "AIzaSyDc0rLG5MxomNV9jZWRFv0B5ke3vXiAIzw" #"YOUR_YOUTUBE_API_KEY"
# channel_id = "UCe2JAC5FUfbxLCfAvBWmNJA" #"USER_PROVIDED_CHANNEL_ID"
# csv_path = "try.csv"
# print(fetch_youtube_data(credentials,api_key, channel_id, csv_path, user_query="paneer makhani"))

# Tokenize and prepare data with mBART
def tokenize_csv(csv_path, tokenizer_path):
    df = pd.read_csv(csv_path)
    tokenizer = MBartTokenizer.from_pretrained(tokenizer_path)
    df = df.fillna("Not available")
    df['Tokenized'] = df['title'] + ' ' + df['description']
    df['Tokenized'] = df['Tokenized'].apply(lambda x: tokenizer.encode(x, return_tensors='pt'))
    return df

# Train or update the recommendation model
def train_model(csv_path, model_path):
    df = pd.read_csv(csv_path)
    df = df.fillna("Not available")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['title'] + ' ' + df['description'])
    
    # Save model and vectorizer
    joblib.dump((tfidf_matrix, vectorizer, df), model_path)

# Recommend recipes based on input
def recommend_recipe(user_input, model_path):
    tfidf_matrix, vectorizer, df = joblib.load(model_path)
    
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, tfidf_matrix)
    
    top_idx = similarities.argsort()[0][-5:][::-1]
    recommended_recipe = df.iloc[top_idx]
    return recommended_recipe

if __name__ == "__main__":
    credentials = authenticate_user()
    api_key = "AIzaSyDc0rLG5MxomNV9jZWRFv0B5ke3vXiAIzw" #"YOUR_YOUTUBE_API_KEY"
    channel_id = "UCe2JAC5FUfbxLCfAvBWmNJA" #"USER_PROVIDED_CHANNEL_ID"
    csv_path = "try.csv"
    MODEL_PATH = "recipe_recommendation_model.pkl"
    TOKENIZER_PATH = "facebook/mbart-large-cc25"
    
    while True:
        user_query = input("Enter recipe query (or 'q' to quit): ")
        if user_query.lower() == 'q':
            print("Exiting...")
            break
        
        # Fetch YouTube data
        query = user_query
        # fetch_youtube_data(credentials,api_key, channel_id, user_query, csv_path)
        videos = get_uploaded_videos(credentials, query=user_query)
        display_videos(videos)
        save_video_dataframe(videos)

        # Tokenize data
        tokenize_csv(csv_path, TOKENIZER_PATH)

        # Train or update recommendation model
        train_model(csv_path, MODEL_PATH)

        # Recommend based on user input
        user_input = user_query
        recommendation = recommend_recipe(user_input, MODEL_PATH)
        print("Recommended Recipe:", recommendation)
