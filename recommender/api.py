from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
from .model import HybridRecommender

app = FastAPI(title="E-commerce Recommender API")

# Load the model and data
recommender = None
original_data = None

class RecommendationRequest(BaseModel):
    user_id: str
    n_items: int = 5

class RecommendationResponse(BaseModel):
    recommendations: List[Dict]
    user_history: Optional[List[Dict]]

@app.on_event("startup")
async def load_model():
    global recommender, original_data
    try:
        # Load the trained model
        recommender = HybridRecommender(use_llm=True)
        recommender.load_model('data/recommender_model.pkl')
        
        # Load original data for product details
        original_data = pd.read_csv(r"C:\Users\nikhi\Downloads\ecommerce_dataset_updated.csv")
        print("Model and data loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
async def root():
    return {"message": "E-commerce Recommender API is running"}

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    if recommender is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Get enhanced recommendations
        enhanced_recs = recommender.recommend_with_explanations(
            user_id=request.user_id,
            n_items=request.n_items
        )
        
        # Get user history
        user_history = original_data[original_data['User_ID'] == request.user_id]
        history_list = []
        
        for _, item in user_history.iterrows():
            history_list.append({
                'category': item['Category'],
                'price': float(item['Price (Rs.)']),
                'discount': float(item['Discount (%)']),
                'final_price': float(item['Final_Price(Rs.)']),
                'purchase_date': item['Purchase_Date']
            })
        
        return {
            "recommendations": enhanced_recs['enhanced_recommendations'],
            "user_history": history_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/enhanced")
async def get_enhanced_recommendations(request: RecommendationRequest):
    if recommender is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    try:
        enhanced_recommendations = recommender.recommend_with_explanations(
            user_id=request.user_id,
            n_items=request.n_items
        )
        
        return enhanced_recommendations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 