import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional
import pickle

class HybridRecommender:
    def __init__(self, use_llm: bool = True, llm_model: str = "llama2"):
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.fitted = False
        
        if use_llm:
            from .llm_agent import OllamaRecommenderAgent
            self.llm_agent = OllamaRecommenderAgent(model_name=llm_model)
    
    def prepare_data(self, interactions_df: pd.DataFrame, item_features_df: pd.DataFrame):
        """Prepare data for training"""
        print("Encoding users and items...")
        self.user_encoder.fit(interactions_df['user_id'])
        self.item_encoder.fit(item_features_df['item_id'])
        
        # Create user-item matrix
        self.user_item_matrix = pd.pivot_table(
            interactions_df,
            values='rating',
            index='user_id',
            columns='item_id',
            fill_value=0
        )
        
        # Process item features
        print("Processing item features...")
        # Encode categorical features
        self.category_encoder.fit(item_features_df['category'])
        category_encoded = pd.get_dummies(
            self.category_encoder.transform(item_features_df['category']),
            prefix='category'
        )
        
        # Normalize numerical features
        scaler = MinMaxScaler()
        numerical_features = item_features_df[['price', 'discount']].copy()
        numerical_features = scaler.fit_transform(numerical_features)
        numerical_features = pd.DataFrame(
            numerical_features,
            columns=['price_normalized', 'discount_normalized']
        )
        
        # Combine features
        processed_features = pd.concat([
            category_encoded,
            numerical_features
        ], axis=1)
        
        processed_features.index = item_features_df['item_id']
        
        return self.user_item_matrix, processed_features
    
    def train(self, user_item_matrix, item_features):
        """Train the model"""
        print("Starting model training...")
        try:
            print(f"User-Item Matrix shape: {user_item_matrix.shape}")
            print(f"Item Features shape: {item_features.shape}")
            
            # Store matrices for recommendations
            self.user_item_matrix = user_item_matrix
            self.item_features = item_features
            
            # Calculate item similarity matrix
            print("Calculating item similarities...")
            self.item_similarity = cosine_similarity(self.item_features)
            
            self.fitted = True
            print("Training completed successfully!")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def recommend(self, user_id: str, n_items: int = 5):
        """Generate recommendations for a user"""
        if not self.fitted:
            raise Exception("Model has not been trained yet!")
        
        # Get user's rated items
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index
        
        # Find similar items to user's rated items
        recommendations = []
        for item in rated_items:
            similar_items = self.item_similarity[
                self.item_features.index.get_loc(item)
            ]
            recommendations.extend(
                list(zip(self.item_features.index, similar_items))
            )
        
        # Sort and filter recommendations
        recommendations = pd.DataFrame(
            recommendations,
            columns=['item_id', 'similarity']
        )
        recommendations = recommendations[
            ~recommendations['item_id'].isin(rated_items)
        ]
        recommendations = recommendations.groupby('item_id')['similarity'].max()
        recommendations = recommendations.sort_values(ascending=False)
        
        return recommendations.index[:n_items].tolist()

    def get_user_history(self, user_id: str) -> pd.DataFrame:
        """Get user's purchase history"""
        # This will be implemented when we connect to the actual dataset
        pass
    
    def get_available_products(self, user_id: str) -> pd.DataFrame:
        """Get available products for recommendation"""
        # This will be implemented when we connect to the actual dataset
        pass
    
    def get_product_details(self, product_id: str) -> pd.Series:
        """Get details for a specific product"""
        # This will be implemented when we connect to the actual dataset
        pass
    
    def find_similar_products(self, product_id: str) -> pd.DataFrame:
        """Find similar products"""
        # This will be implemented when we connect to the actual dataset
        pass
    
    def recommend_with_explanations(self, user_id: str, n_items: int = 5) -> Dict:
        """Generate LLM-powered recommendations"""
        # Get user history
        user_history = self.get_user_history(user_id)
        
        # Analyze user preferences using LLM
        preferences = self.llm_agent.analyze_user_preferences(user_history)
        
        # Get available products
        available_products = self.get_available_products(user_id)
        
        # Generate recommendations using LLM
        recommendations = self.llm_agent.generate_recommendations(
            preferences,
            available_products,
            n_items
        )
        
        # Generate detailed explanations for each recommendation
        enhanced_recommendations = []
        for rec in recommendations:
            product_id = rec['product_id']
            product = self.get_product_details(product_id)
            similar_products = self.find_similar_products(product_id)
            
            explanation = self.llm_agent.explain_recommendation(
                product,
                preferences,
                similar_products
            )
            
            enhanced_recommendations.append({
                "product_id": product_id,
                "reasoning": rec['reasoning'],
                "explanation": explanation['explanation'],
                "value_proposition": explanation['value_prop'],
                "alternatives": explanation['alternatives'],
                "considerations": explanation['considerations']
            })
        
        return {
            "user_preferences": preferences,
            "enhanced_recommendations": enhanced_recommendations
        }
    
    def _get_user_data(self, user_id: str) -> Dict:
        """Get user profile data"""
        # Placeholder - implement based on your user data storage
        return {
            "user_id": user_id,
            "total_interactions": len(self._get_user_history(user_id))
        }
    
    def _get_user_history(self, user_id: str) -> List[Dict]:
        """Get user interaction history"""
        # Placeholder - implement based on your data storage
        return []
    
    def _get_item_data(self, item_id: str) -> Dict:
        """Get item details"""
        # Placeholder - implement based on your item data storage
        return {
            "item_id": item_id,
            "category": "unknown",
            "brand": "unknown",
            "price": 0.0
        }

    def save_model(self, path: str):
        """Save the model to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'user_encoder': self.user_encoder,
                'item_encoder': self.item_encoder,
                'category_encoder': self.category_encoder,
                'fitted': self.fitted
            }, f)

    @classmethod
    def load_model(cls, path: str) -> 'HybridRecommender':
        """Load the model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls()
        instance.model = data['model']
        instance.user_encoder = data['user_encoder']
        instance.item_encoder = data['item_encoder']
        instance.category_encoder = data['category_encoder']
        instance.fitted = data['fitted']
        
        return instance 