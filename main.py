from fastapi import FastAPI  # type: ignore
from pydantic import BaseModel
import sqlite3
from sqlite3 import Error
from datetime import datetime
from transformers import pipeline
from fastapi.responses import JSONResponse  # type: ignore
import os

app = FastAPI()

# Use the 3-label sentiment analysis model from Hugging Face
sentiment_analyzer = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment')

DATABASE = "sentiment_results.db"

# Helper function to create a database connection
def create_connection():
    try:
        return sqlite3.connect(DATABASE)
    except Error as e:
        print(f"Connection Error: {e}")
        return None

# Function to create a table in the database
def create_table():
    conn = create_connection()
    if conn:
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    sentiment_label TEXT NOT NULL,
                    sentiment_score REAL NOT NULL,
                    timestamp TEXT NOT NULL
                );
            """)
            conn.commit()
        except Error as e:
            print(f"Table Creation Error: {e}")
        finally:
            conn.close()

# Function to check and create the database if it does not exist
def check_and_create_db():
    if not os.path.exists(DATABASE):
        create_table()
    try:
        conn = sqlite3.connect(DATABASE)
        conn.close()
    except sqlite3.DatabaseError:
        os.remove(DATABASE)
        create_table()

# Check and create the database at startup
check_and_create_db()

# Pydantic model to define request structure
class SentimentRequest(BaseModel):
    text: str

# Mapping of model labels to desired output labels
LABEL_MAPPING = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

# Endpoint for sentiment analysis prediction
@app.post("/predict/")
async def analyze_sentiment(request: SentimentRequest):
    try:
        # Perform sentiment analysis on the input text
        sentiment_result = sentiment_analyzer(request.text)[0]
        label = sentiment_result['label']
        score = sentiment_result['score']
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Map the model's label to "positive", "neutral", or "negative"
        mapped_label = LABEL_MAPPING.get(label, "unknown")

        # Store the sentiment result in the database
        conn = create_connection()
        if conn:
            conn.execute(
                "INSERT INTO sentiments (text, sentiment_label, sentiment_score, timestamp) VALUES (?, ?, ?, ?)",
                (request.text, mapped_label, score, timestamp)
            )
            conn.commit()
            conn.close()

        # Return the mapped sentiment label and score
        return {"label": mapped_label, "score": score}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

# Endpoint to fetch sentiment analysis history from the database
@app.get("/history/")
async def get_sentiment_history():
    conn = create_connection()
    if conn:
        try:
            cursor = conn.execute("SELECT * FROM sentiments")
            rows = cursor.fetchall()
            sentiment_data = [
                {
                    "id": row[0],
                    "text": row[1],
                    "sentiment_label": row[2],
                    "sentiment_score": row[3],
                    "timestamp": row[4]
                }
                for row in rows
            ]
            return {"sentiments": sentiment_data}
        except Error as e:
            return JSONResponse(status_code=500, content={"message": str(e)})
        finally:
            conn.close()

# Start the FastAPI app
if __name__ == "__main__":
    import uvicorn # type: ignore
    uvicorn.run(app, host="0.0.0.0", port=8000)
