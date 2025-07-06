from typing import Union
from fastapi import FastAPI
from pyngrok import ngrok
import nest_asyncio
from pyngrok import ngrok
import uvicorn
# model.py
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from fastapi.middleware.cors import CORSMiddleware

# Connect and fetch data from MongoDB
client = MongoClient('mongodb+srv://rakshit186005:2Q6e9BL5eotTfE3o@authdata.1vqhzsu.mongodb.net/AuthData')

def fetch_user_data():
    db = client['AuthData']
    collection = db['userwatchlists']
    cursor = collection.find({})
    df = pd.DataFrame(list(cursor))
    return df

# Process data into matrices
def prepare_matrices(df):
    df_exploded = df.explode('Watchlist')
    watchlist_df = pd.json_normalize(df_exploded['Watchlist'])
    final_df = pd.concat([df_exploded[['email', '_id']].reset_index(drop=True), watchlist_df], axis=1)
    final_df['media_type'].fillna('movie', inplace=True)
    final_df['user_rating'] = final_df['user_rating'].replace(['neutral','positive','negative'], [5,8,3])
    final_df['rating_encoded'] = final_df['user_rating']

    prof_df = final_df.explode('genre_id')
    prof_mat = prof_df.pivot_table(index='email', columns=['media_type', 'original_language', 'user_rating', 'genre_id'], fill_value=0, aggfunc='size')
    ucf = final_df.pivot_table(index='email', columns=['movie_id'], values='rating_encoded')
    return final_df, prof_mat, ucf

# Compute cosine similarity matrices
def compute_similarity_matrices(ucf, prof_mat):
    cf_sim = cosine_similarity(ucf.fillna(0))
    norm_prof = normalize(prof_mat)
    profile_sim = cosine_similarity(norm_prof)
    return pd.DataFrame(cf_sim, index=ucf.index, columns=ucf.index), pd.DataFrame(profile_sim, index=prof_mat.index, columns=prof_mat.index)

# Blend similarities with alpha
def blend_similarities(cf_sim, profile_sim, alpha=0.3):
    return alpha * cf_sim + (1 - alpha) * profile_sim

# Recommend using weighted projected ratings
def recommend_from_neighbors(user_email, top_sim_users, ucf, top_k=10):
    user_movies = set(ucf.loc[user_email].dropna().index)
    candidate_scores = {}
    sim_sums = {}

    for sim_user, sim_score in top_sim_users.items():
        sim_user_ratings = ucf.loc[sim_user].dropna()
        for movie, rating in sim_user_ratings.items():
            if movie not in user_movies:
                candidate_scores[movie] = candidate_scores.get(movie, 0) + sim_score * rating
                sim_sums[movie] = sim_sums.get(movie, 0) + sim_score

    projected_ratings = {
        movie: candidate_scores[movie] / sim_sums[movie]
        for movie in candidate_scores if sim_sums[movie] > 0
    }

    sorted_recs = sorted(projected_ratings.items(), key=lambda x: x[1], reverse=True)
    return sorted_recs[:top_k]

# Final function to call from FastAPI
def get_recommendations(email, top_k=15, alpha=0.3):
    df = fetch_user_data()
    final_df, prof_mat, ucf = prepare_matrices(df)

    if email not in ucf.index:
        return {
            "similar_user": None,
            "recommendations": []
        }

    cf_sim, profile_sim = compute_similarity_matrices(ucf, prof_mat)
    final_sim = blend_similarities(cf_sim, profile_sim, alpha=alpha)

    sims = final_sim.loc[email].drop(email).sort_values(ascending=False)
    top_sim_users = sims.head(10)
    
    final_df.drop(columns='_id')
    
    recs = recommend_from_neighbors(email, top_sim_users, ucf, top_k=top_k)
    recommended_df = final_df[final_df['movie_id'].isin([m for m, _ in recs])]
    recommended_df = recommended_df.drop(columns=["_id","email"], errors="ignore")
    scores = {m: s for m, s in recs}

    recommended_df['projected_rating'] = recommended_df['movie_id'].map(scores)
    recommended_df.sort_values(by='projected_rating', ascending=False, inplace=True)
    

    return {
        "data": recommended_df.to_dict(orient='records')
    }

nest_asyncio.apply()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/recommend/{email}")
def recommend(email: str, top_k: Union[int, None] = 15):
    return get_recommendations(email, top_k=top_k)
