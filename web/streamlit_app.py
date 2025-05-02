import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from googletrans import Translator
from textblob import TextBlob
from streamlit_echarts import st_echarts
from typing import Dict
from streamlit_chat import message
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from recommender.llm_agent import HuggingFaceRecommenderAgent

# Initialize translator
@st.cache_resource
def get_translator():
    return Translator()

# Initialize the recommender
@st.cache_resource
def load_recommender():
    return HuggingFaceRecommenderAgent()

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\nikhi\Downloads\ecommerce_dataset_updated.csv")
    
    # Convert dates with correct format (DD-MM-YYYY)
    df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], format='%d-%m-%Y')
    
    # Translate non-English categories
    translator = get_translator()
    unique_categories = df['Category'].unique()
    translations = {}
    for cat in unique_categories:
        try:
            if not TextBlob(cat).detect_language() == 'en':
                translations[cat] = translator.translate(cat, dest='en').text
        except:
            translations[cat] = cat
    
    df['Category_EN'] = df['Category'].map(translations).fillna(df['Category'])
    return df

def create_spending_trend(data):
    monthly_spend = data.groupby(data['Purchase_Date'].dt.to_period('M')).agg({
        'Final_Price(Rs.)': 'sum',
        'Product_ID': 'count'
    }).reset_index()
    monthly_spend['Purchase_Date'] = monthly_spend['Purchase_Date'].astype(str)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_spend['Purchase_Date'],
        y=monthly_spend['Final_Price(Rs.)'],
        name='Total Spend',
        line=dict(color='#2E86C1', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=monthly_spend['Purchase_Date'],
        y=monthly_spend['Product_ID'] * monthly_spend['Final_Price(Rs.)'].mean(),
        name='Purchase Volume',
        line=dict(color='#28B463', width=3, dash='dot')
    ))
    
    fig.update_layout(
        title='Monthly Shopping Trends',
        xaxis_title='Month',
        yaxis_title='Amount (₹)',
        template='plotly_white',
        hovermode='x unified'
    )
    return fig

def create_price_discount_scatter(data):
    fig = px.scatter(
        data,
        x='Price (Rs.)',
        y='Discount (%)',
        color='Category_EN',
        size='Final_Price(Rs.)',
        hover_data=['Purchase_Date'],
        title='Price vs Discount Analysis',
        template='plotly_white'
    )
    fig.update_layout(
        xaxis_title='Original Price (₹)',
        yaxis_title='Discount %',
        showlegend=True
    )
    return fig

# Modern UI Configuration
st.set_page_config(
    page_title="Smart Shop AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern UI Theme
st.markdown("""
    <style>
    /* Modern Color Scheme */
    :root {
        --primary-color: #7C3AED;
        --secondary-color: #4F46E5;
        --background-color: #F3F4F6;
        --card-color: #FFFFFF;
        --text-color: #1F2937;
        --accent-color: #10B981;
    }
    
    /* Global Styles */
    .main {
        background-color: var(--background-color);
        padding: 2rem;
    }
    
    /* Modern Header */
    .header-container {
        background: linear-gradient(120deg, var(--primary-color), var(--secondary-color));
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Dashboard Cards */
    .dashboard-card {
        background: var(--card-color);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #E5E7EB;
    }
    
    .dashboard-card:hover {
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    /* Metric Cards */
    .metric-container {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }
    
    .metric-card {
        background: var(--card-color);
        padding: 1.5rem;
        border-radius: 12px;
        flex: 1;
        min-width: 200px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--primary-color);
    }
    
    .metric-label {
        color: #6B7280;
        font-size: 0.9rem;
    }
    
    /* Chart Container */
    .chart-container {
        background: var(--card-color);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: var(--card-color);
        padding: 0.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        color: var(--text-color);
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: var(--primary-color);
        color: white;
    }
    
    /* Recommendations */
    .recommendation-card {
        background: var(--card-color);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid var(--accent-color);
    }
    
    /* Progress Bars */
    .stProgress > div > div {
        background-color: var(--accent-color);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: var(--card-color);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: var(--secondary-color);
        border: none;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--background-color);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 4px;
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 1.5s infinite;
    }
    
    .recommendation-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .recommendation-header {
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .assistant-message {
        background: #f8f9fa;
        border-left: 4px solid var(--primary-color);
        padding: 1rem;
        margin-top: 0.5rem;
        border-radius: 0 8px 8px 0;
        font-style: italic;
        color: #4b5563;
    }
    
    .recommendation-details {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 2rem;
    }
    
    .product-info {
        flex: 2;
    }
    
    .match-score {
        flex: 1;
        text-align: center;
    }
    
    .score-circle {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem auto;
    }
    
    .product-info p {
        margin: 0.5rem 0;
        color: #4b5563;
    }
    
    .product-info strong {
        color: #1f2937;
    }
    
    /* Main text colors */
    .main {
        color: #1a1a1a;
    }
    
    /* Card text colors */
    .recommendation-card,
    .dashboard-card,
    .metric-card,
    .chart-container {
        color: #1a1a1a;
    }
    
    /* AI message styling */
    .assistant-message {
        background-color: #f8f9fa;
        border-left: 4px solid var(--primary-color);
        padding: 1rem;
        margin-top: 0.5rem;
        border-radius: 0 8px 8px 0;
        color: #1a1a1a;
        font-style: italic;
    }
    
    /* Metric values */
    .metric-value {
        color: #1a1a1a;
        font-weight: 600;
    }
    
    .metric-label {
        color: #4a5568;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a !important;
    }
    
    /* Product details */
    .product-info p {
        color: #1a1a1a;
        margin: 0.5rem 0;
    }
    
    /* AI insights box */
    div[style*='background-color: white'] {
        color: #1a1a1a !important;
    }
    
    /* Analysis text */
    div[style*='background-color: #f8f9fa'] {
        color: #1a1a1a !important;
    }
    
    /* Response boxes */
    div[style*='background-color: #f0f9ff'] {
        color: #1a1a1a !important;
    }
    
    /* Links */
    a {
        color: var(--primary-color) !important;
    }
    
    /* Strong text */
    strong {
        color: #1a1a1a;
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        color: #1a1a1a !important;
    }
    
    /* Table text */
    .dataframe {
        color: #1a1a1a !important;
    }
    
    /* Sidebar text */
    .css-1d391kg {
        color: #1a1a1a;
    }
    
    /* Input labels */
    .stSelectbox label,
    .stSlider label,
    .stTextInput label {
        color: #1a1a1a !important;
    }
    
    /* Chart labels */
    .js-plotly-plot text {
        color: #1a1a1a !important;
    }
    
    /* Chat UI Styles */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 20px;
        border-radius: 15px;
        background: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .chat-message {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 15px;
        border-radius: 10px;
        max-width: 80%;
    }
    
    .chat-message.user {
        background: #EEF2FF;
        margin-left: auto;
        flex-direction: row-reverse;
    }
    
    .chat-message.assistant {
        background: #F3F4F6;
        margin-right: auto;
    }
    
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }
    
    .user .avatar {
        background: #4F46E5;
        color: white;
    }
    
    .assistant .avatar {
        background: #10B981;
        color: white;
    }
    
    .message-content {
        padding: 8px;
    }
    
    .quick-actions {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 10px;
        margin-top: 20px;
    }
    
    .action-button {
        background: white;
        border: 1px solid #E5E7EB;
        padding: 10px;
        border-radius: 8px;
        text-align: left;
        transition: all 0.3s ease;
    }
    
    .action-button:hover {
        background: #F3F4F6;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

def create_header():
    st.markdown("""
        <div class="header-container">
            <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">🛍️ Smart Shop AI</h1>
            <p style="font-size: 1.1rem; opacity: 0.9;">Intelligent Shopping Analysis & Recommendations</p>
        </div>
    """, unsafe_allow_html=True)

def create_metrics(user_history):
    st.markdown("""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-value">₹{:,.2f}</div>
                <div class="metric-label">Total Spent</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Items Purchased</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{:.1f}%</div>
                <div class="metric-label">Average Discount</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Categories Explored</div>
            </div>
        </div>
    """.format(
        user_history['Final_Price(Rs.)'].sum(),
        len(user_history),
        user_history['Discount (%)'].mean(),
        user_history['Category'].nunique()
    ), unsafe_allow_html=True)

def create_interactive_timeline(user_history):
    # Create a figure using Plotly Express
    fig = px.line(
        user_history,
        x='Purchase_Date',
        y='Final_Price(Rs.)',
        title='Shopping Trends Over Time',
        labels={
            'Purchase_Date': 'Date',
            'Final_Price(Rs.)': 'Amount Spent (₹)'
        }
    )
    
    # Update layout
    fig.update_layout(
        hovermode='x unified',
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#f0f0f0'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#f0f0f0'
        )
    )
    
    return fig

def create_interactive_filters(data):
    col1, col2 = st.columns(2)
    with col1:
        date_range = st.date_input(
            "Select Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            key="date_filter"
        )
    with col2:
        price_range = st.slider(
            "Price Range (₹)",
            0, 10000, (0, 10000),
            key="price_filter"
        )
    
    categories = st.multiselect(
        "Select Categories",
        ["All Categories"] + sorted(data['Category_EN'].unique().tolist()),
        default=["All Categories"],
        key="category_filter"
    )
    
    return date_range, price_range, categories

def create_interactive_product_comparison(recommendations, data):
    if not recommendations:
        return
    
    # Get product details for comparison
    products = []
    for rec in recommendations:
        product = data[data['Product_ID'] == rec['product_id']].iloc[0]
        products.append({
            'id': rec['product_id'],
            'price': product['Price (Rs.)'],
            'discount': product['Discount (%)'],
            'category': product['Category_EN'],
            'final_price': product['Final_Price(Rs.)']
        })
    
    # Create a proper Plotly figure from the comparison data
    comparison_fig = go.Figure(
        data=[
            go.Bar(
                name="Original Price",
                x=[str(p['id']) for p in products],
                y=[p['price'] for p in products]
            ),
            go.Bar(
                name="Final Price",
                x=[str(p['id']) for p in products],
                y=[p['final_price'] for p in products]
            ),
            go.Scatter(
                name="Discount",
                x=[str(p['id']) for p in products],
                y=[p['discount'] for p in products],
                yaxis='y2'
            )
        ]
    )
    
    # Update layout
    comparison_fig.update_layout(
        title="Product Comparison",
        xaxis_title="Product ID",
        yaxis_title="Price (₹)",
        yaxis2=dict(
            title="Discount (%)",
            overlaying='y',
            side='right'
        ),
        barmode='group',
        hovermode='x unified'
    )
    
    return comparison_fig

# Add this function to handle chat responses
def get_chatbot_response(user_input: str, preferences: dict, history: pd.DataFrame) -> str:
    # Detect intent from user input
    intents = {
        'recommendations': ['recommend', 'suggest', 'similar', 'like'],
        'analysis': ['analyze', 'pattern', 'insight', 'trend'],
        'deals': ['deal', 'discount', 'offer', 'save'],
        'product_info': ['tell me about', 'what is', 'details', 'info'],
        'comparison': ['compare', 'difference', 'versus', 'vs']
    }
    
    detected_intent = None
    for intent, keywords in intents.items():
        if any(keyword in user_input.lower() for keyword in keywords):
            detected_intent = intent
            break
    
    # Enhance prompt based on intent
    context = f"""
    User Profile:
    - Preferred categories: {', '.join(preferences.get('preferred_categories', []))}
    - Average spend: ₹{history['Final_Price(Rs.)'].mean():.2f}
    - Typical discount range: {history['Discount (%)'].mean():.1f}%
    
    Recent Purchases:
    {history[['Category_EN', 'Final_Price(Rs.)', 'Purchase_Date']].tail(3).to_string()}
    
    Intent: {detected_intent if detected_intent else 'general'}
    Question: {user_input}
    """
    
    try:
        response = agent.generate_response(context)
        
        # Format response based on intent
        if detected_intent == 'recommendations':
            response += "\n\n💡 Would you like to see more specific recommendations in any particular category?"
        elif detected_intent == 'analysis':
            response += "\n\n📊 I can provide more detailed analytics if you're interested in a specific aspect."
        elif detected_intent == 'deals':
            response += "\n\n💰 I can also set up alerts for deals in your preferred categories."
            
        return response
    except Exception as e:
        return "I apologize, but I'm having trouble processing your request. Could you please try rephrasing it?"

def render_chat_message(message, key):
    avatar_icon = "👤" if message["role"] == "user" else "🤖"
    st.markdown(f"""
        <div class="chat-message {message['role']}">
            <div class="avatar">{avatar_icon}</div>
            <div class="message-content">{message['content']}</div>
        </div>
    """, unsafe_allow_html=True)

def main():
    # Initialize
    agent = load_recommender()
    data = load_data()
    
    create_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div style="padding: 1rem 0;">
                <h2 style="color: var(--primary-color);">📊 Analysis Controls</h2>
            </div>
        """, unsafe_allow_html=True)
        
        user_id = st.selectbox(
            "Select User ID",
            options=sorted(data['User_ID'].unique()),
            key="user_select"
        )
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.markdown("""
            <div style="padding: 1rem 0;">
                <h3 style="color: var(--text-color);">🎯 Quick Stats</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.metric("Total Users", f"{data['User_ID'].nunique():,}")
        st.metric("Product Categories", f"{data['Category'].nunique()}")
    
    if user_id:
        user_history = data[data['User_ID'] == user_id]
        
        # Metrics
        create_metrics(user_history)
        
        # Main Content Tabs
        tabs = st.tabs([
            "📊 Shopping Analysis",
            "🎯 AI Insights",
            "💡 Recommendations",
            "🤖 AI Shopping Assistant",
            "🔄 Product Comparison"
        ])
        
        with tabs[0]:
            st.markdown("### 📈 Shopping Timeline")
            timeline_fig = create_interactive_timeline(user_history)
            st.plotly_chart(timeline_fig, use_container_width=True)
        
        with tabs[1]:
            with st.spinner("Analyzing shopping patterns..."):
                preferences = agent.analyze_user_preferences(user_history)
                
                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                st.json(preferences)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tabs[2]:
            with st.spinner("Generating smart recommendations..."):
                available_products = data[~data['User_ID'].isin([user_id])].head(20)
                recommendations = agent.generate_recommendations(
                    preferences,
                    available_products,
                    n_items=5
                )
                
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"""
                        <div class="recommendation-card">
                            <h3>Recommendation {i}</h3>
                            <p><strong>Product ID:</strong> {rec['product_id']}</p>
                            <p><strong>Reasoning:</strong> {rec['reasoning']}</p>
                        </div>
                    """, unsafe_allow_html=True)
        
        # New AI Shopping Assistant tab
        with tabs[3]:
            st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
            
            # Display chat messages
            for idx, message in enumerate(st.session_state.messages):
                render_chat_message(message, idx)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Enhanced quick actions
            st.markdown("### 💡 Quick Actions")
            quick_actions = {
                "📊 Analyze Shopping Patterns": "Can you analyze my shopping patterns and give me insights?",
                "🎯 Get Personalized Recommendations": "What products would you recommend based on my history?",
                "💰 Find Best Deals": "What are the best deals available for me right now?",
                "📈 Price Trends": "Can you show me price trends for my favorite categories?",
                "🔍 Similar Products": "Show me products similar to my recent purchases"
            }
            
            st.markdown('<div class="quick-actions">', unsafe_allow_html=True)
            for action_label, action_prompt in quick_actions.items():
                if st.button(action_label, key=f"action_{action_label}"):
                    st.session_state.messages.append({"role": "user", "content": action_prompt})
                    with st.spinner("Thinking..."):
                        response = get_chatbot_response(action_prompt, preferences, user_history)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tabs[4]:
            with st.spinner("Generating product comparison..."):
                available_products = data[~data['User_ID'].isin([user_id])].head(2)
                comparison_fig = create_interactive_product_comparison(recommendations, data)
                st.plotly_chart(comparison_fig, use_container_width=True)
    
    else:
        st.info("👈 Select a user ID from the sidebar to start the analysis")

    # Install required package
    st.markdown("""
        ```bash
        pip install streamlit-chat
        ```
    """)

if __name__ == "__main__":
    main()