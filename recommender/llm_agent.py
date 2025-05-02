from transformers import pipeline
import json
from typing import Dict, List
import pandas as pd

class HuggingFaceRecommenderAgent:
    def __init__(self):
        self.model = pipeline(
            "text2text-generation",
            model="facebook/bart-large",
            device=-1  # Use CPU
        )
        self.max_length = 512
        self.min_length = 50
        
    def generate_response(self, prompt: str) -> str:
        """Generate response using the model"""
        try:
            response = self.model(
                prompt,
                max_length=self.max_length,
                min_length=self.min_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )[0]['generated_text']
            return response
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now."
    
    def analyze_user_preferences(self, user_history: pd.DataFrame) -> Dict:
        """Analyze user preferences using the model"""
        prompt = f"""
        Analyze this customer's shopping preferences based on their history:
        
        Purchase History:
        {user_history[['Category_EN', 'Final_Price(Rs.)', 'Discount (%)']].to_string()}
        
        Provide a structured analysis of:
        1. Preferred categories
        2. Price range preferences
        3. Discount sensitivity
        4. Shopping patterns
        """
        
        analysis = self.generate_response(prompt)
        
        # Parse the response into structured data
        try:
            preferences = {
                'preferred_categories': user_history['Category_EN'].value_counts().index.tolist()[:3],
                'price_range': f"₹{user_history['Final_Price(Rs.)'].mean():.2f} average",
                'shopping_patterns': analysis,
                'discount_preference': f"{user_history['Discount (%)'].mean():.1f}% average discount"
            }
        except Exception as e:
            preferences = {
                'preferred_categories': [],
                'price_range': 'Not enough data',
                'shopping_patterns': 'Not enough data',
                'discount_preference': 'Not enough data'
            }
            
        return preferences
    
    def get_chatbot_response(self, user_input: str, preferences: Dict, history: pd.DataFrame) -> str:
        """Generate chatbot response using the same model"""
        prompt = f"""
        You are a helpful AI shopping assistant. Respond to this customer query:

        Customer Profile:
        - Preferred categories: {', '.join(preferences.get('preferred_categories', []))}
        - Price range: {preferences.get('price_range', 'Not specified')}
        - Shopping patterns: {preferences.get('shopping_patterns', 'Not specified')}
        
        Recent Purchase History:
        {history[['Category_EN', 'Final_Price(Rs.)', 'Discount (%)']].tail(3).to_string()}

        Customer Question: {user_input}

        Provide a helpful, friendly response that:
        1. Directly answers their question
        2. Uses their shopping history for context
        3. Gives specific product recommendations when relevant
        4. Maintains a conversational, friendly tone
        """
        
        return self.generate_response(prompt)
    
    def generate_recommendations(self, 
                               user_preferences: Dict,
                               available_products: pd.DataFrame,
                               n_items: int = 5) -> List[Dict]:
        """Generate product recommendations using the same model"""
        recommendations = []
        
        for _, product in available_products.head(n_items).iterrows():
            prompt = f"""
            Analyze this product's suitability for the customer:
            
            Customer Preferences:
            - Preferred categories: {', '.join(user_preferences['preferred_categories'])}
            - Price range: {user_preferences['price_range']}
            - Shopping patterns: {user_preferences['shopping_patterns']}
            
            Product Details:
            - Category: {product['Category_EN']}
            - Price: ₹{product['Price (Rs.)']:,.2f}
            - Discount: {product['Discount (%)']}%
            - Final Price: ₹{product['Final_Price(Rs.)']:,.2f}
            
            Explain why this product would be suitable for this customer.
            """
            
            reasoning = self.generate_response(prompt)
            
            recommendations.append({
                "product_id": str(product['Product_ID']),
                "reasoning": reasoning
            })
        
        return recommendations

    def generate_personalized_message(self, user_preferences: Dict, product: pd.Series) -> str:
        """Generate a personalized shopping recommendation message"""
        prompt = f"""
        As a personal shopping assistant, create a friendly, personalized message (2-3 sentences) for this customer.

        Customer Profile:
        - Preferred categories: {', '.join(user_preferences['preferred_categories'])}
        - Typical price range: {user_preferences['price_range']}
        - Shopping pattern: {user_preferences['shopping_patterns']}

        Product:
        - Category: {product['Category_EN']}
        - Price: ₹{product['Price (Rs.)']:,.2f}
        - Discount: {product['Discount (%)']}%
        - Final Price: ₹{product['Final_Price(Rs.)']:,.2f}

        Write a natural, conversational recommendation explaining why this product matches their preferences.
        Focus on their shopping patterns and preferences. Make it personal and engaging.
        """
        
        try:
            message = self.generate_response(prompt)
            return message
        except Exception as e:
            return f"This {product['Category_EN']} matches your shopping preferences and is available at a {product['Discount (%)']}% discount!"

    def explain_recommendation(self,
                             product: pd.Series,
                             user_preferences: Dict,
                             similar_products: pd.DataFrame) -> Dict:
        """Generate detailed explanation for a recommendation"""
        explanation_prompt = f"""
        Explain why this product matches the user's preferences:
        
        Product Details:
        - Category: {product['Category']}
        - Price: ₹{product['Price (Rs.)']}
        - Discount: {product['Discount (%)']}%
        
        User Preferences:
        {json.dumps(user_preferences, indent=2)}
        
        Similar Products Available:
        {similar_products[['Category', 'Price (Rs.)', 'Discount (%)']].to_string()}
        
        Provide:
        1. Main reason for recommendation
        2. Value proposition
        3. Alternative suggestions
        4. Special considerations
        
        Format response as JSON with keys: explanation, value_prop, alternatives, considerations
        """
        
        response = self.generate_response(explanation_prompt)
        try:
            return json.loads(response)
        except:
            return {
                "explanation": "Error generating explanation",
                "value_prop": "Unknown",
                "alternatives": [],
                "considerations": []
            } 