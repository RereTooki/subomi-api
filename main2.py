from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import uvicorn
import pandas as pd
import bcrypt
import pickle
import pandas as pd
import pyshorteners
import sys  
from urllib.parse import quote
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sqlite3
from googleapiclient.discovery import build
from fastapi.middleware.cors import CORSMiddleware

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



app = FastAPI(title="Student Performance Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=False,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

API_KEY = 'AIzaSyC3-_W6tAJgww_E6Btzp3nseeoFIqdOTlM'

file_path = 'Books.csv'
data = pd.read_csv(file_path)

data['Book-Title'] = data['Book-Title'].str.lower()

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['Book-Title'])


def search_youtube(course_name, max_results=5):
    query = f"everything i need to know on {course_name}"
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    request = youtube.search().list(
        q=query,          
        part='snippet', 
        maxResults=max_results, 
        type='video'        
    )
    response = request.execute()
    videos = []
    for item in response['items']:
        video_id = item['id']['videoId']
        channel = item['snippet']['channelTitle']
        url = f"https://www.youtube.com/watch?v={video_id}"
        videos.append((url, channel))
    return videos

def recommend_books(subject_title, top_n=5):
    query_vector = vectorizer.transform([subject_title.lower()])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[-top_n:][::-1]
    recommended_books = data.iloc[similar_indices].copy()    
    return recommended_books[['Book-Title', 'Book-Author', 'Publisher', 'Year-Of-Publication']]



with open(os.path.join(BASE_DIR, "Ridge_Regressor_Second_Model.pkl"), "rb") as file:
    potential_score_ridge_model = pickle.load(file)
print(potential_score_ridge_model)


with open(os.path.join(BASE_DIR, "Lasso_Regressor_Second_Model.pkl"), "rb") as file:
    potential_score_lasso_model = pickle.load(file)
print(potential_score_lasso_model)

with open(os.path.join(BASE_DIR, "Ridge_Regressor_Studyhours_Model.pkl"), "rb") as file:
    study_hours_ridge_model = pickle.load(file)
print(study_hours_ridge_model)


with open(os.path.join(BASE_DIR, "Lasso_Regressor_Studyhours_Model.pkl"), "rb") as file:
    study_hours_lasso_model = pickle.load(file)
print(study_hours_lasso_model)

conn = sqlite3.connect('student_performance_tesinnnnnnt_1565.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT,
    password TEXT NOT NULL,
    extracurricular_activities INTEGER
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS courses (
    course_id INTEGER PRIMARY KEY,
    user_id INTEGER,
    course_name TEXT NOT NULL,
    hours_for_lecture INTEGER,
    learning_type INTEGER,
    difficulty_level INTEGER,
    predicted_performance REAL,
    study_time REAL,
    publisher TEXT,
    textbooks TEXT,
    youtube_URL TEXT,
    advice TEXT,
    FOREIGN KEY (user_id) REFERENCES users (user_id)
)
''')

conn.commit()
print("Database and tables created successfully.")

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

def is_email_registered(email):
    cursor.execute("SELECT COUNT(*) FROM users WHERE email = ?", (email,))
    result = cursor.fetchone()[0]
    return result > 0

def add_user(name, email, password):
    cursor.execute('INSERT INTO users (name, email, password, extracurricular_activities) VALUES (?, ?, ?, ?)', 
                  (name, email, password, 0))
    conn.commit()
    return cursor.lastrowid


def login(email, password):
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    if user and verify_password(password, user[3]): 
        print(f"Welcome back, {user[1]}!")
        return user
    else:
        print("Invalid email or password. Please try again.")
        return None
    

def add_course(user_id, course_name, hours_for_lecture, learning_type, difficulty_level, predicted_performance, study_time, publisher, textbooks, youtube_URL, advice):    
    if isinstance(publisher, pd.Series):
        publisher = ', '.join(publisher.tolist())    
    if isinstance(textbooks, pd.Series):
        textbooks = ', '.join(textbooks.tolist()) 
    if isinstance(youtube_URL, pd.Series):
        youtube_URL = ', '.join(youtube_URL.tolist())
    cursor.execute('''
    INSERT INTO courses (user_id, course_name, hours_for_lecture, learning_type, difficulty_level, predicted_performance, study_time, publisher, textbooks, youtube_URL, advice)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, course_name, hours_for_lecture, learning_type, difficulty_level, predicted_performance, study_time, publisher, textbooks, youtube_URL, advice))
    conn.commit()
    print(f"Course '{course_name}' added for user ID {user_id}.")


def get_and_print_user_courses(user_id):
    cursor.execute('''
    SELECT course_name, hours_for_lecture, learning_type, difficulty_level, predicted_performance, study_time, publisher, textbooks, youtube_URL, advice
    FROM courses WHERE user_id = ?
    ''', (user_id,))
    courses_data = cursor.fetchall()
    if courses_data:
        columns = ['Course Name', 'Hours for Lecture', 'Learning Type', 'Difficulty Level', 'Predicted Performance', 'study_time', 'publisher', 'textbooks', 'youtube_URL', 'advice']
        courses_df = pd.DataFrame(courses_data, columns=columns)
        print("\nCourses for User ID:", user_id)
        print(courses_df.to_string(index=False))
    else:
        print("No courses found for this user.")


def main():
    name = input("Enter your name: ")
    email = input("Enter your email: ")
    if is_email_registered(email):
        print("This email is already registered. Please login.")
        sys.exit()
    password = input("Enter your password: ")
    extracurricular_activities = int(input("How many Extracurricular Activities do you have: "))
    user_id = add_user(name, email, password, extracurricular_activities)

    has_general_learning_type = input("Do you have a general learning type? (yes/no): ").strip().lower()

    overall_learning_type = None
    if has_general_learning_type == 'yes':
        learning_type = input("Enter your learning type: ")
        if learning_type == 'textual': 
            overall_learning_type = 1
        elif learning_type == 'visual':
            overall_learning_type = 2
        elif learning_type == 'interactive': 
            overall_learning_type = 3
        elif learning_type == 'textual and visual':
            overall_learning_type = 4
        elif learning_type == 'textual and interactive':
            overall_learning_type = 5
        elif learning_type == 'visual and interactive':
            overall_learning_type = 6
        elif learning_type == 'textual, visual and interactive':
            overall_learning_type = 7
        else:
            print("Invalid learning type")

    while True:
        course_name = input("Enter course name (or type 'stop' to finish): ")
        if course_name.lower() == 'stop':
            break
        hours_for_lecture = int(input("Enter hours of lecture per week: "))
        
        if overall_learning_type is not None:
            learning_type = overall_learning_type
        else:
            string_learning_type = input("Enter your learning type: ")
            if string_learning_type == 'textual': 
                learning_type = 1
            elif string_learning_type == 'visual':
                learning_type = 2
            elif string_learning_type == 'interactive': 
                learning_type = 3
            elif string_learning_type == 'textual and visual':
                learning_type = 4
            elif string_learning_type == 'textual and interactive':
                learning_type = 5
            elif string_learning_type == 'visual and interactive':
                learning_type = 6
            elif string_learning_type == 'textual, visual and interactive':
                learning_type = 7
            else:
                print("Invalid learning type")
        
        difficulty_level = int(input("Enter difficulty level (1-10): "))

        new_data_predicted_peformance = pd.DataFrame({
            'hours_for_lecture': [hours_for_lecture],
            'extracurricular': [extracurricular_activities],
            'learning_type': [learning_type],
            'difficulty_level': [difficulty_level]
        })
        extracurricular = new_data_predicted_peformance['extracurricular'].values[0]
        difficulty = new_data_predicted_peformance['difficulty_level'].values[0]
        
        new_ridge_performance_predictions = potential_score_ridge_model.predict(new_data_predicted_peformance)
        new_lasso__performance_predictions = potential_score_lasso_model.predict(new_data_predicted_peformance)
        final_prediction_performance_index = np.round((new_ridge_performance_predictions + new_lasso__performance_predictions) / 2)
        predicted_performance = float(final_prediction_performance_index[0])
        
        new_data_predicted_study_hours = pd.DataFrame({
            'credit_unit': [hours_for_lecture],
            'potential_score_in_subject': final_prediction_performance_index
        })
        
        credit_unit = new_data_predicted_study_hours['credit_unit'].values[0]
        potential_score_in_subject = new_data_predicted_study_hours['potential_score_in_subject'].values[0]
        
        for i in range(0,3):
            new_ridge_predictions = study_hours_ridge_model.predict(new_data_predicted_study_hours)
            new_lasso_predictions = study_hours_lasso_model.predict(new_data_predicted_study_hours)
            final_prediction_study_hours = (new_ridge_predictions + new_lasso_predictions) / 2
        rounded_predictions = np.round(final_prediction_study_hours / 11, 1)
        study_time = float(rounded_predictions[0])

        recommendations = recommend_books(course_name, top_n=5)
        
        textbooks = recommendations['Book-Title'].reset_index(drop=True)
        publisher = recommendations['Publisher'].reset_index(drop=True)
        youtube_URL = search_youtube(course_name)
        youtube_URL = "\n".join([f"- {url} ({channel})" for url, channel in youtube_URL])
        
        if final_prediction_performance_index[0] < 70:
            advice = f"\n[AI Analysis]: Based on your inputs, your predicted score in this course is {final_prediction_performance_index[0]}. To optimize your performance, consider the following adjustments:"
            if extracurricular > 2 and difficulty > 5:
                advice += f"\n- **Suggested Action**: Reduce extracurricular activities to allocate more focus on this challenging course."
            if learning_type < 5 and difficulty > 5:
                advice += f"\n- **Learning Style Optimization**: Adjust your learning strategy to align better with the course's high difficulty level."
            if extracurricular > 2 and difficulty < 6:
                advice += f"\n- **Action Recommended**: Consider reducing extracurriculars slightly to allow more dedicated time for studying."
            if learning_type < 5 and difficulty < 6:
                advice += f"\n- **Suggested Learning Type**: Try incorporating textbooks, YouTube videos, and group study sessions for a balanced approach."
            if difficulty > 7 and learning_type > 5 and extracurricular < 2:
                advice += f"\n[AI Observation]: This course appears exceptionally challenging. Consider discussing with your instructor for additional support."
            advice += f"\n**Weekly Study Hours**: Based on AI calculations, studying this course for approximately {rounded_predictions[0]} hours weekly should help improve your predicted score."
        else:
            advice = f"\n[AI Analysis]: Your predicted score in this course is {final_prediction_performance_index[0]}, indicating strong potential for a top grade."
            advice += f"\n**Weekly Study Hours**: To maintain or exceed this performance, dedicating {rounded_predictions[0]} hours per week is recommended."
            
        textbooks_with_publishers = "\n".join(
            f"- **{title}** by {pub}" for title, pub in zip(textbooks, publisher)
        )
        youtube_videos_list = youtube_URL

        if learning_type == 1:
            advice += f"\n\n[Learning Style-Based Recommendations]: As a primarily textual learner, AI suggests the following textbooks: \n{textbooks_with_publishers}"
            advice += f"\n*Supplementary Visual Resources*: You might find additional insights from these YouTube videos: \n{youtube_videos_list}"
        elif learning_type == 2:
            advice += f"\n\n[Learning Style-Based Recommendations]: Since you're a visual learner, AI recommends focusing on these YouTube videos: \n{youtube_videos_list}"
            advice += f"\n*Supplementary Textual Resources*: Textbooks you might find useful: \n{textbooks_with_publishers}"
        elif learning_type == 3:
            advice += "\n\n[Learning Style-Based Recommendations]: As an interactive learner, AI suggests these strategies to maximize engagement:"
            advice += """
            \n1. **Group Discussions**: Share ideas in study groups.
            \n2. **Interactive Tools**: Use flashcards, quizzes, or similar.
            \n3. **Real-Life Applications**: Simulate scenarios to test your knowledge.
            \n4. **Q&A Practice**: Engage in projects with peers or mentors.
            """
            advice += f"\nAdditionally, AI recommends these textbooks: \n{textbooks_with_publishers} and YouTube resources: \n{youtube_videos_list}"
        elif learning_type == 4:
            advice += f"\n\n[Learning Style-Based Recommendations]: As both a textual and visual learner, these resources should be beneficial:\n- Textbooks: \n{textbooks_with_publishers}\n- YouTube Videos: \n{youtube_videos_list}"
        elif learning_type == 5:
            advice += f"\n\n[Learning Style-Based Recommendations]: As a textual and interactive learner, AI recommends these textbooks: \n{textbooks_with_publishers}"
            advice += "\n**Enhanced Learning Tips**:"
            advice += """
            \n1. **Group Discussions**: Share and learn through peer sessions.
            \n2. **Interactive Tools**: Incorporate flashcards or quizzes.
            \n3. **Scenario Simulations**: Apply concepts practically.
            \n4. **Project-Based Learning**: Regular Q&A sessions can aid comprehension.
            """
            advice += f"\nAdditionally, AI recommends these YouTube resources: \n{youtube_videos_list}"
        elif learning_type == 6:
            advice += f"\n\n[Learning Style-Based Recommendations]: As a visual and interactive learner, the following YouTube videos are recommended: \n{youtube_videos_list}"
            advice += "\n**Additional Learning Suggestions**:"
            advice += """
            \n1. **Group Discussions**: Engage actively in study groups.
            \n2. **Interactive Tools**: Use flashcards or quizzes for reinforcement.
            \n3. **Practical Applications**: Implement real-world examples.
            \n4. **Project Engagement**: Focus on hands-on projects for deeper understanding.
            """
            advice += f"\nAdditionally, AI recommends these textbooks: \n{textbooks_with_publishers}"
        elif learning_type == 7:
            advice += f"\n\n[Learning Style-Based Recommendations]: Given your textual, visual, and interactive learning preferences, AI suggests the following resources:\n- Textbooks: \n{textbooks_with_publishers}\n- YouTube Videos: \n{youtube_videos_list}"
            advice += "\n**Interactive Learning Tips**:"
            advice += """
            \n1. **Group Discussions**: Collaborate in study groups.
            \n2. **Interactive Tools**: Enhance retention with flashcards and quizzes.
            \n3. **Scenario-Based Learning**: Apply knowledge to simulated situations.
            \n4. **Hands-On Projects**: Q&A sessions and projects boost learning.
            """
        add_course(user_id, course_name, hours_for_lecture, learning_type, difficulty_level, predicted_performance, study_time, publisher, textbooks, youtube_URL, advice)

    get_and_print_user_courses(user_id)

    email_login = input("Enter your email to login: ")
    password_login = input("Enter your password: ")
    value = login(email_login, password_login)





#### BRO i MADE CHANGES FROM HERE DOWN-------------------------------------------------------------------------------------------------------------------

# I didnt do the fifth endpoint because the fourth one already returns the advice, so we don't need a personalized endpoint to do that.




# Pydantic Models
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserExtracurricular(BaseModel):
    user_id: int
    name: str
    extracurricular_activities: int

class CourseCreate(BaseModel):
    course_name: str
    hours_for_lecture: int
    learning_type: int
    difficulty_level: int
    extracurricular_activities: int

class CourseResponse(BaseModel):
    course_name: str
    hours_for_lecture: int
    learning_type: int
    difficulty_level: int
    predicted_performance: float
    study_time: float
    publisher: str
    textbooks: str
    youtube_URL: str
    advice: str

class UserDashboard(BaseModel):
    user_id: int
    name: str
    courses: List[dict]
    overall_stats: dict

class PredictionResult(BaseModel):
    course_name: str
    predicted_grade: float
    study_hours_per_week: float
    personalized_advice: str

# Endpoints
@app.post("/register", response_model=dict)
async def register_user(user: UserCreate):
    if is_email_registered(user.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash the password before storing
    hashed_password = hash_password(user.password)
    
    user_id = add_user(
        user.name,
        user.email,
        hashed_password  # Store the hashed password instead of plain text
    )
    return {"message": "User registered successfully", "user_id": user_id}

@app.post("/login", response_model=dict)
async def login_user(user: UserLogin):
    cursor.execute("SELECT * FROM users WHERE email = ?", (user.email,))
    user_data = cursor.fetchone()
    
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Verify the hashed password
    if not verify_password(user.password, user_data[3]):  # user_data[3] contains the hashed password
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    return {
        "message": "Login successful",
        "user_id": user_data[0],
        "name": user_data[1]
    }

@app.get("/users/{user_id}/extracurricular", response_model=UserExtracurricular)
async def get_user_extracurricular(user_id: int):
    cursor.execute("SELECT user_id, name, extracurricular_activities FROM users WHERE user_id = ?", (user_id,))
    user_data = cursor.fetchone()
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    return UserExtracurricular(
        user_id=user_data[0],
        name=user_data[1],
        extracurricular_activities=user_data[2]
    )


# @app.post("/personalized_predictions")

@app.post("/courses", response_model=CourseResponse)
async def create_course(course: CourseCreate, user_id: int):
    # Get recommendations
    recommendations = recommend_books(course.course_name, top_n=5)
    textbooks = recommendations['Book-Title'].reset_index(drop=True)
    publisher = recommendations['Publisher'].reset_index(drop=True)
    youtube_URL = search_youtube(course.course_name)
    youtube_URL = "\n".join([f"- {url} ({channel})" for url, channel in youtube_URL])

    # Convert Pandas Series to strings
    publisher_str = publisher.to_string(index=False)
    textbooks_str = textbooks.to_string(index=False)

    update_user_extracurricular(user_id, course.extracurricular_activities)

    new_data_predicted_peformance = pd.DataFrame({
        'hours_for_lecture': [course.hours_for_lecture],
        'extracurricular': [course.extracurricular_activities],
        'learning_type': [course.learning_type],
        'difficulty_level': [course.difficulty_level]
    })

    extracurricular = new_data_predicted_peformance['extracurricular'].values[0]
    difficulty = new_data_predicted_peformance['difficulty_level'].values[0]
    learning_type = new_data_predicted_peformance['learning_type'].values[0]

    new_ridge_performance_predictions = potential_score_ridge_model.predict(new_data_predicted_peformance)
    new_lasso__performance_predictions = potential_score_lasso_model.predict(new_data_predicted_peformance)
    final_prediction_performance_index = np.round((new_ridge_performance_predictions + new_lasso__performance_predictions) / 2)
    predicted_performance = float(final_prediction_performance_index[0])
    
    new_data_predicted_study_hours = pd.DataFrame({
        'credit_unit': [course.hours_for_lecture],
        'potential_score_in_subject': final_prediction_performance_index
    })
    
    for i in range (0,3):
        new_ridge_predictions = study_hours_ridge_model.predict(new_data_predicted_study_hours)
        new_lasso_predictions = study_hours_lasso_model.predict(new_data_predicted_study_hours)
        final_prediction_study_hours = (new_ridge_predictions + new_lasso_predictions) / 2
    rounded_predictions = np.round(final_prediction_study_hours / 11, 1)
    study_time = float(rounded_predictions[0])

    # Calculate predictions (simplified version - you'll need to implement the full logic)
    if final_prediction_performance_index[0] < 70:
        advice = f"\n[AI Analysis]: Based on your inputs, your predicted score in this course is {final_prediction_performance_index[0]}. To optimize your performance, consider the following adjustments:"
        if extracurricular > 2 and difficulty > 5:
            advice += f"\n- **Suggested Action**: Reduce extracurricular activities to allocate more focus on this challenging course."
        if learning_type < 5 and difficulty > 5:
            advice += f"\n- **Learning Style Optimization**: Adjust your learning strategy to align better with the course's high difficulty level."
        if extracurricular > 2 and difficulty < 6:
            advice += f"\n- **Action Recommended**: Consider reducing extracurriculars slightly to allow more dedicated time for studying."
        if learning_type < 5 and difficulty < 6:
            advice += f"\n- **Suggested Learning Type**: Try incorporating textbooks, YouTube videos, and group study sessions for a balanced approach."
        if difficulty > 7 and learning_type > 5 and extracurricular < 2:
            advice += f"\n[AI Observation]: This course appears exceptionally challenging. Consider discussing with your instructor for additional support."
        advice += f"\n**Weekly Study Hours**: Based on AI calculations, studying this course for approximately {rounded_predictions[0]} hours weekly should help improve your predicted score."
    else:
        advice = f"\n[AI Analysis]: Your predicted score in this course is {final_prediction_performance_index[0]}, indicating strong potential for a top grade."
        advice += f"\n**Weekly Study Hours**: To maintain or exceed this performance, dedicating {rounded_predictions[0]} hours per week is recommended."
        
    textbooks_with_publishers = "\n".join(
        f"- **{title}** by {pub}" for title, pub in zip(textbooks, publisher)
    )
    youtube_videos_list = youtube_URL

    if learning_type == 1:
        advice += f"\n\n[Learning Style-Based Recommendations]: As a primarily textual learner, AI suggests the following textbooks: \n{textbooks_with_publishers}"
        advice += f"\n*Supplementary Visual Resources*: You might find additional insights from these YouTube videos: \n{youtube_videos_list}"
    elif learning_type == 2:
        advice += f"\n\n[Learning Style-Based Recommendations]: Since you're a visual learner, AI recommends focusing on these YouTube videos: \n{youtube_videos_list}"
        advice += f"\n*Supplementary Textual Resources*: Textbooks you might find useful: \n{textbooks_with_publishers}"
    elif learning_type == 3:
        advice += "\n\n[Learning Style-Based Recommendations]: As an interactive learner, AI suggests these strategies to maximize engagement:"
        advice += """
        \n1. **Group Discussions**: Share ideas in study groups.
        \n2. **Interactive Tools**: Use flashcards, quizzes, or similar.
        \n3. **Real-Life Applications**: Simulate scenarios to test your knowledge.
        \n4. **Q&A Practice**: Engage in projects with peers or mentors.
        """
        advice += f"\nAdditionally, AI recommends these textbooks: \n{textbooks_with_publishers} and YouTube resources: \n{youtube_videos_list}"
    elif learning_type == 4:
        advice += f"\n\n[Learning Style-Based Recommendations]: As both a textual and visual learner, these resources should be beneficial:\n- Textbooks: \n{textbooks_with_publishers}\n- YouTube Videos: \n{youtube_videos_list}"
    elif learning_type == 5:
        advice += f"\n\n[Learning Style-Based Recommendations]: As a textual and interactive learner, AI recommends these textbooks: \n{textbooks_with_publishers}"
        advice += "\n**Enhanced Learning Tips**:"
        advice += """
        \n1. **Group Discussions**: Share and learn through peer sessions.
        \n2. **Interactive Tools**: Incorporate flashcards or quizzes.
        \n3. **Scenario Simulations**: Apply concepts practically.
        \n4. **Project-Based Learning**: Regular Q&A sessions can aid comprehension.
        """
        advice += f"\nAdditionally, AI recommends these YouTube resources: \n{youtube_videos_list}"
    elif learning_type == 6:
        advice += f"\n\n[Learning Style-Based Recommendations]: As a visual and interactive learner, the following YouTube videos are recommended: \n{youtube_videos_list}"
        advice += "\n**Additional Learning Suggestions**:"
        advice += """
        \n1. **Group Discussions**: Engage actively in study groups.
        \n2. **Interactive Tools**: Use flashcards or quizzes for reinforcement.
        \n3. **Practical Applications**: Implement real-world examples.
        \n4. **Project Engagement**: Focus on hands-on projects for deeper understanding.
        """
        advice += f"\nAdditionally, AI recommends these textbooks: \n{textbooks_with_publishers}"
    elif learning_type == 7:
        advice += f"\n\n[Learning Style-Based Recommendations]: Given your textual, visual, and interactive learning preferences, AI suggests the following resources:\n- Textbooks: \n{textbooks_with_publishers}\n- YouTube Videos: \n{youtube_videos_list}"
        advice += "\n**Interactive Learning Tips**:"
        advice += """
        \n1. **Group Discussions**: Collaborate in study groups.
        \n2. **Interactive Tools**: Enhance retention with flashcards and quizzes.
        \n3. **Scenario-Based Learning**: Apply knowledge to simulated situations.
        \n4. **Hands-On Projects**: Q&A sessions and projects boost learning.
        """

    # Add course to database
    add_course(
        user_id,
        course.course_name,
        course.hours_for_lecture,
        course.learning_type,
        course.difficulty_level,
        predicted_performance,
        study_time,
        publisher_str,
        textbooks_str,
        youtube_URL,
        advice
    )

    return CourseResponse(
        course_name=course.course_name,
        hours_for_lecture=course.hours_for_lecture,
        learning_type=course.learning_type,
        difficulty_level=course.difficulty_level,
        predicted_performance=predicted_performance,
        study_time=study_time,
        publisher=publisher_str,
        textbooks=textbooks_str,
        youtube_URL=youtube_URL,
        advice=advice
    )

def get_user_courses(user_id):
    cursor.execute('''
    SELECT course_name, hours_for_lecture, learning_type, difficulty_level, predicted_performance, study_time, publisher, textbooks, youtube_URL, advice
    FROM courses WHERE user_id = ?
    ''', (user_id,))
    courses_data = cursor.fetchall()
    if courses_data:
        courses = []
        for course in courses_data:
            course_response = CourseResponse(
                course_name=course[0],
                hours_for_lecture=course[1],
                learning_type=course[2],
                difficulty_level=course[3],
                predicted_performance=course[4],
                study_time=course[5],
                publisher=course[6],
                textbooks=course[7],
                youtube_URL=course[8],
                advice=course[9]
            )
            courses.append(course_response)
        return courses
    return []

@app.get("/courses/{user_id}", response_model=List[CourseResponse])
async def get_user_courses_endpoint(user_id: int):
    courses = get_user_courses(user_id)
    if not courses:
        raise HTTPException(status_code=404, detail="No courses found for this user")
    return courses

@app.get("/recommendations/{course_name}")
async def get_recommendations(course_name: str):
    books = recommend_books(course_name)
    videos = search_youtube(course_name)
    return {
        "books": books.to_dict(orient='records'),
        "videos": videos
    }

@app.get("/users/{user_id}/recommendations/{course_name}")
async def get_user_course_recommendations(user_id: int, course_name: str):
    # Get user information
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    user_data = cursor.fetchone()
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get user's course information
    cursor.execute("""
        SELECT course_name, hours_for_lecture, learning_type, difficulty_level, 
               predicted_performance, study_time, publisher, textbooks, youtube_URL, advice
        FROM courses 
        WHERE user_id = ? AND course_name = ?
        ORDER BY course_id DESC 
        LIMIT 1
    """, (user_id, course_name))
    course_data = cursor.fetchone()
    
    if not course_data:
        raise HTTPException(status_code=404, detail=f"No course found for user {user_id} with course name {course_name}")
    
    # Convert the course data to a dictionary
    course_info = {
        "course_name": course_data[0],
        "hours_for_lecture": course_data[1],
        "learning_type": course_data[2],
        "difficulty_level": course_data[3],
        "predicted_performance": course_data[4],
        "study_time": course_data[5],
        "publisher": course_data[6],
        "textbooks": course_data[7],
        "youtube_URL": course_data[8],
        "advice": course_data[9]
    }
    
    return course_info['advice']

def update_user_extracurricular(user_id: int, extracurricular_activities: int):
    cursor.execute("""
        UPDATE users 
        SET extracurricular_activities = ? 
        WHERE user_id = ?
    """, (extracurricular_activities, user_id))
    conn.commit()
    return cursor.rowcount > 0

@app.get("/users/{user_id}/dashboard", response_model=UserDashboard)
async def get_user_dashboard(user_id: int):
    # Get user information
    cursor.execute("SELECT user_id, name FROM users WHERE user_id = ?", (user_id,))
    user_data = cursor.fetchone()
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get all courses for the user
    cursor.execute("""
        SELECT course_name, predicted_performance, study_time, publisher, textbooks, youtube_URL, advice
        FROM courses 
        WHERE user_id = ?
        ORDER BY course_id DESC
    """, (user_id,))
    courses_data = cursor.fetchall()
    
    if not courses_data:
        raise HTTPException(status_code=404, detail="No courses found for this user")
    
    # Process courses data
    courses = []
    total_performance = 0
    total_study_hours = 0
    
    for course in courses_data:
        course_info = {
            "course_name": course[0],
            "predicted_grade": course[1],
            "weekly_study_hours": course[2],
            "resources": {
                "textbooks": course[4].split('\n') if course[4] else [],
                "videos": course[5].split('\n') if course[5] else [],
                "publishers": course[3].split('\n') if course[3] else []
            },
            "advice": course[6]
        }
        courses.append(course_info)
        total_performance += course[1]
        total_study_hours += course[2]
    
    # Calculate overall statistics
    num_courses = len(courses)
    overall_stats = {
        "average_predicted_grade": round(total_performance / num_courses, 2),
        "total_weekly_study_hours": round(total_study_hours, 1),
        "number_of_courses": num_courses,
        "performance_trend": "Improving" if courses[0]["predicted_grade"] > courses[-1]["predicted_grade"] else "Needs Attention" if courses[0]["predicted_grade"] < 70 else "Stable"
    }
    
    return UserDashboard(
        user_id=user_data[0],
        name=user_data[1],
        courses=courses,
        overall_stats=overall_stats
    )

@app.get("/users/{user_id}/predictions/{course_name}", response_model=PredictionResult)
async def get_prediction_results(user_id: int, course_name: str):
    # Get user information
    cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
    user_data = cursor.fetchone()
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get prediction results for the specific course
    cursor.execute("""
        SELECT course_name, predicted_performance, study_time, advice
        FROM courses 
        WHERE user_id = ? AND course_name = ?
        ORDER BY course_id DESC
        LIMIT 1
    """, (user_id, course_name))
    prediction_data = cursor.fetchone()
    
    if not prediction_data:
        raise HTTPException(status_code=404, detail=f"No predictions found for course {course_name}")
    
    return PredictionResult(
        course_name=prediction_data[0],
        predicted_grade=prediction_data[1],
        study_hours_per_week=prediction_data[2],
        personalized_advice=prediction_data[3]
    )

if __name__ == "__main__":
    # main()
    uvicorn.run(app, host="0.0.0.0", port=8000)
