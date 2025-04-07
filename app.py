import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import requests
import json
import os
import random
import sqlite3
import joblib
from typing import Dict
from collections import Counter
import google.generativeai as genai

# MUST BE THE VERY FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Career Roadmap AI", layout="wide")

# Replace with your actual Gemini API key
GEMINI_API_KEY = "AIzaSyC4wYOTzTyNkdOQuFrlRI2nofPV955OVCM"  # Replace with your actual key from https://ai.google.dev/
genai.configure(api_key=GEMINI_API_KEY)

# 1. DATA COLLECTION AND SIMULATION
def simulate_career_data(num_samples=1000):
    """Simulate realistic career data with specific skill categories"""
    roles = {
        'Software Developer': {'Tech': (3, 5), 'Soft Skills': (2, 4)},
        'Data Scientist': {'Tech': (4, 5), 'Soft Skills': (2, 4), 'Analytical': (3, 5)},
        'Project Manager': {'Business': (3, 5), 'Soft Skills': (4, 5), 'Tech': (2, 4)},
        'UX Designer': {'Tech': (3, 5), 'Soft Skills': (3, 5), 'Design': (3, 5)},
        'Marketing Specialist': {'Marketing': (4, 5), 'Soft Skills': (3, 5), 'Analytical': (2, 4)}
    }

    career_paths = [
        ('Junior', 'Mid-Level', 'Senior', 'Lead'),
        ('Analyst', 'Specialist', 'Manager', 'Director')
    ]

    exp_levels = range(1, 16)
    data = []
    skill_categories = ['Tech', 'Marketing', 'Business', 'Soft Skills']

    for _ in range(num_samples):
        path_idx = random.randint(0, len(career_paths) - 1)
        career_path = career_paths[path_idx]
        exp = random.choice(exp_levels)
        current_level = min(int(exp / 4), len(career_path) - 1)
        target_level = min(current_level + 1, len(career_path) - 1)
        current_role_base = random.choice(list(roles.keys()))

        profile = {'Current Role': f"{career_path[current_level]} {current_role_base}",
                   'Target Role': f"{career_path[target_level]} {current_role_base}",
                   'Experience': exp}

        for skill_cat in skill_categories:
            min_val = 1
            max_val = 3
            for role, skills in roles.items():
                if role in current_role_base and skill_cat in skills:
                    min_val, max_val = skills[skill_cat]
                    break
            current_val = random.randint(1, max_val)
            target_val = random.randint(current_val, 5)
            profile[f'{skill_cat}_Current'] = current_val
            profile[f'{skill_cat}_Target'] = target_val

        data.append(profile)

    return pd.DataFrame(data)

# 2. DATA PREPROCESSING AND MODEL TRAINING
def preprocess_and_train(data, force_retrain=False):
    """Preprocess data and train model, save for real-time use"""
    model_file = 'knn_model_specific_axes.pkl'
    scaler_file = 'scaler_specific_axes.pkl'
    skill_categories = ['Tech', 'Marketing', 'Business', 'Soft Skills']

    if os.path.exists(model_file) and not force_retrain:
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
    else:
        X_columns = [f'{skill}_Current' for skill in skill_categories] + ['Experience']
        X = data[X_columns].fillna(0).values
        y_columns = [f'{skill}_Target' for skill in skill_categories]
        y = data[y_columns].fillna(0).values

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)
        print(f"Specific Axes Model R² score: {score:.2f}")

        joblib.dump(model, model_file)
        joblib.dump(scaler, scaler_file)

    return model, scaler, skill_categories

# 3. DATABASE FOR PERSISTENCE
def init_db():
    """Initialize SQLite database for user data"""
    conn = sqlite3.connect('user_data_specific_axes.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, current_role TEXT, target_role TEXT,
                  experience INTEGER, skills TEXT, roadmap TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()

def save_user_data(current_role, target_role, experience, skills, roadmap):
    """Save user input and roadmap to database"""
    conn = sqlite3.connect('user_data_specific_axes.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (current_role, target_role, experience, skills, roadmap, timestamp) VALUES (?, ?, ?, ?, ?, datetime('now'))",
              (current_role, target_role, experience, json.dumps(skills), roadmap))
    conn.commit()
    conn.close()

# 4. ROADMAP GENERATION
def get_roadmap(current_role, target_role, experience, current_skills, target_skills):
    """Generates a structured roadmap of Coursera courses based on user data."""
    short_term = f"Build foundational skills for {current_role} based on current skills: {', '.join([f'{k}: {v}' for k, v in current_skills.items()])}"
    mid_term = f"Develop intermediate skills to transition from {current_role} to {target_role} with experience of {experience} years"
    long_term = f"Achieve expertise for {target_role} with target skills: {', '.join([f'{k}: {v:.1f}' for k, v in target_skills.items()])}"

    prompt = f"""
Analyze the following tasks and create a detailed, structured roadmap of Coursera courses to complete them, similar to the structure found on roadmap.sh.

Organize the roadmap into sections: "Short-Term Goals", "Mid-Term Goals", and "Long-Term Goals".

For each goal, provide:
- A clear title for the goal.
- A list of Coursera courses with direct, clickable links in Markdown format (e.g., [Course Name](Course URL)).
- A brief description of how each course helps achieve the goal.
- Order the courses in a logical learning sequence.

Limit the response to about 700 words.

Short-term tasks: {short_term}
Mid-term tasks: {mid_term}
Long-term tasks: {long_term}

Output Structure:
## Short-Term Goals
- *Goal Title:*
    - [Course Name](Course URL): Description
    - [Course Name](Course URL): Description
- *Goal Title:*
    - ...

## Mid-Term Goals
- *Goal Title:*
    - ...

## Long-Term Goals
- *Goal Title:*
    - ...
"""
    response_content = get_api_response(prompt)
    return parse_human_response(response_content)

def get_api_response(prompt):
    """Fetches response from Gemini API."""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')  # Adjust model name as needed
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error fetching Gemini API response: {e}")
        return f"Error: Unable to generate roadmap due to API issue - {str(e)}"

def parse_human_response(response_content):
    """Parses the response to make it sound more human-like."""
    if "Error:" in response_content:
        return response_content
    response_content = response_content.replace("<br>", "").replace("<br/>", "").replace("<br />", "").replace("*", "")
    unwanted_phrases = [
        "Okay, let's start:\n\n",
        "Alright, here we go:\n\n",
        "Let's begin with:\n\n"
    ]
    for phrase in unwanted_phrases:
        response_content = response_content.replace(phrase, "")
    return response_content

# 5. STREAMLIT REAL-TIME APP WITH ENHANCED FRONTEND
def create_streamlit_app(model, scaler, skill_names):
    """Real-time Streamlit application with an attractive roadmap frontend"""
    # Custom CSS for a visually appealing roadmap
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f0f2f5, #e1e6ea);
        padding: 30px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333;
    }
    .title-container {
        background-color: #263959;
        color: white;
        padding: 25px 0;
        border-radius: 10px;
        margin-bottom: 40px;
        text-align: center;
    }
    .title {
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 1.2em;
        color: #ddd;
    }
    .form-container {
        background-color: white;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        margin-bottom: 40px;
        border-left: 5px solid #5cb85c; /* Accent color */
    }
    .st-subheader {
        color: #263959;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSlider>div>div>div>div[data-baseweb="slider"] {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
        font-size: 16px;
    }
    .stSlider>div>div>div>div[data-baseweb="slider"] {
        padding: 5px;
    }
    .stButton>button {
        background-color: #5cb85c;
        color: white;
        border: none;
        padding: 15px 30px;
        font-size: 18px;
        border-radius: 25px;
        cursor: pointer;
        transition: background-color 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #4cae4c;
    }
    .roadmap-header {
        color: #263959;
        font-size: 2em;
        font-weight: bold;
        margin-bottom: 20px;
        text-align: center;
    }
    .roadmap-section {
        background-color: #f9f9f9;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
        border-left: 5px solid #007bff; /* Another accent color */
    }
    .roadmap-section h2 {
        color: #007bff;
        font-size: 1.8em;
        margin-top: 0;
        margin-bottom: 15px;
        padding-bottom: 5px;
        border-bottom: 2px solid #007bff;
    }
    .goal {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
        border-left: 3px solid #ffc107; /* Another accent */
    }
    .goal h3 {
        color: #333;
        font-size: 1.4em;
        margin-top: 0;
        margin-bottom: 10px;
    }
    .course {
        margin-left: 20px;
        margin-bottom: 10px;
        font-size: 1em;
    }
    .course a {
        color: #007bff;
        text-decoration: none;
        font-weight: 500;
    }
    .course a:hover {
        text-decoration: underline;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #eee;
        color: #777;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="title-container"><h1 class="title">🚀 Career Roadmap AI</h1><p class="subtitle">Plan your journey to your dream career</p></div>', unsafe_allow_html=True)

    # Input Form
    with st.container():
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        with st.form(key='user_input_form'):
            st.subheader("Tell Us About Your Career Aspirations")
            current_role = st.text_input("Current Job Role", "e.g., Software Developer")
            target_role = st.text_input("Dream Job Role", "e.g., Marketing Manager")
            experience_input = st.slider("Years of Experience", 0, 40, 5)

            st.subheader("Your Current Skill Set")
            skills_input_text = st.text_area("List your current skills (comma-separated)", value="Python, Communication, Project Management", height=100)

            submit_button = st.form_submit_button(label="✨ Generate My Personalized Roadmap ✨")
        st.markdown('</div>', unsafe_allow_html=True)

    skill_categories_for_prediction = ['Tech', 'Marketing', 'Business', 'Soft Skills']

    if submit_button:
        input_skills = [skill.strip().lower() for skill in skills_input_text.split(",") if skill.strip()]

        # Initialize current skill levels based on user input
        current_skills_input = {skill: 1 for skill in skill_categories_for_prediction}
        for skill in skill_categories_for_prediction:
            for user_skill in input_skills:
                if skill.lower() in user_skill:
                    current_skills_input[skill] = max(current_skills_input[skill], 3)

        user_input = [current_skills_input.get(skill, 1) for skill in skill_categories_for_prediction] + [experience_input]
        user_input_scaled = scaler.transform([user_input])
        predicted_skills = model.predict(user_input_scaled)[0]
        target_skills = {skill: min(round(pred), 5) for skill, pred in zip(skill_categories_for_prediction, predicted_skills)}

        # Generate roadmap using Gemini API
        with st.spinner("🧠 Crafting your personalized roadmap with AI..."):
            roadmap = get_roadmap(current_role, target_role, experience_input, current_skills_input, target_skills)
            save_user_data(current_role, target_role, experience_input, current_skills_input, roadmap)

        # Display Roadmap
        st.markdown('<h2 class="roadmap-header">Your Personalized Career Roadmap 🗺</h2>', unsafe_allow_html=True)

        # Split roadmap into sections and display
        sections = roadmap.split("## ")[1:]  # Skip empty first split
        for section in sections:
            section_title, content = section.split("\n", 1)
            st.markdown(f'<div class="roadmap-section"><h2>{section_title}</h2>{content}</div>', unsafe_allow_html=True)

    # Footer
    st.markdown('<div class="footer">Powered by xAI & Gemini AI</div>', unsafe_allow_html=True)

# MAIN EXECUTION
def main():
    """Main function for real-time app"""
    data_file = 'career_data_specific_axes.csv'
    if not os.path.exists(data_file):
        st.write("Simulating career data for specific axes...")
        career_data = simulate_career_data()
        career_data.to_csv(data_file, index=False)
    else:
        career_data = pd.read_csv(data_file)

    # Initialize the database for persistence
    init_db()

    # Preprocess the data and train (or load) the model and scaler
    model, scaler, skill_names = preprocess_and_train(career_data)

    # Run the Streamlit app
    create_streamlit_app(model, scaler, skill_names)

if __name__ == "__main__":
    main()
