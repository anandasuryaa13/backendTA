from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS (untuk React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load semua objek dari file .joblib (pastikan sudah menyimpan train_df)
(
    user_item_matrix,
    item_user_matrix,
    brand_user_matrix,
    le_item,
    le_user,
    le_brand,
    train_df
) = joblib.load("knn_model_assets.joblib")

# Buat mapping dari nama barang ke kode barang
name_to_code = dict(zip(train_df['NAMABARA'], train_df['KODEBARA']))

class Req(BaseModel):
    selected_items: List[str]
    preference_type: str  # 'item', 'user', 'brand'

@app.post("/recommend")
def recommend_route(req: Req):
    preference_type = req.preference_type

    # Ubah nama barang ke kode barang
    selected_item_codes = [name_to_code.get(nama) for nama in req.selected_items if name_to_code.get(nama)]
    if not selected_item_codes:
        return {"recommendations": []}

    encoded_items = le_item.transform(selected_item_codes)
    result_series = pd.Series(dtype='float64')

    if preference_type == 'user':
        users = train_df[train_df['KODEBARA_encoded'].isin(encoded_items)]['NAMA_encoded'].unique()
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(user_item_matrix)

        for user in users:
            try:
                distances, indices = model.kneighbors([user_item_matrix.loc[user]], n_neighbors=6)
                neighbors = indices.flatten()[1:]
                similar_users = user_item_matrix.iloc[neighbors]
                result_series = result_series.add(similar_users.mean(axis=0), fill_value=0)
            except:
                continue

    elif preference_type == 'brand':
        brands = train_df[train_df['KODEBARA_encoded'].isin(encoded_items)]['BRAND_encoded'].unique()
        users = train_df[train_df['BRAND_encoded'].isin(brands)]['NAMA_encoded'].unique()
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(brand_user_matrix)

        for user in users:
            try:
                distances, indices = model.kneighbors([brand_user_matrix.loc[user]], n_neighbors=6)
                neighbors = indices.flatten()[1:]
                similar_users = brand_user_matrix.iloc[neighbors]
                brand_scores = similar_users.mean(axis=0)
                top_brands = brand_scores.sort_values(ascending=False).index[:5]
                items_in_top_brand = train_df[train_df['BRAND_encoded'].isin(top_brands)]['KODEBARA_encoded'].value_counts()
                result_series = result_series.add(items_in_top_brand, fill_value=0)
            except:
                continue

    else:  # item-based
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(item_user_matrix)
        for item in encoded_items:
            try:
                distances, indices = model.kneighbors([item_user_matrix.loc[item]], n_neighbors=6)
                neighbors = indices.flatten()[1:]
                similar_items = item_user_matrix.iloc[neighbors].mean(axis=1)
                result_series = result_series.add(similar_items, fill_value=0)
            except:
                continue

        result_series = result_series.drop(labels=encoded_items, errors='ignore')
    top_items = result_series.sort_values(ascending=False).head(5).index
    recommended_codes = le_item.inverse_transform(np.array(top_items, dtype=int)).tolist()

    # Gabungkan kode + nama barang
    recommended_full = []
    for kode in recommended_codes:
        row = train_df[train_df['KODEBARA'] == kode]
        if not row.empty:
            nama = row.iloc[0]['NAMABARA']
            recommended_full.append(f"{kode} - {nama}")

    return {"recommendations": recommended_full}


@app.get("/items")
def get_items():
    # Kirim daftar nama barang ke frontend
    all_items = sorted(train_df['NAMABARA'].unique().tolist())
    return {"items": all_items}

@app.get("/brands")
def get_brands():
    brands = sorted(train_df['BRAND'].unique().tolist())
    return {"brands": brands}
