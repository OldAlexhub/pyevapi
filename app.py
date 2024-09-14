import numpy as np
import pandas as pd
from prophet import Prophet
from flask_cors import CORS
import pymongo
from flask import Flask
import os

app = Flask(__name__)
CORS(app)

# Conditionally load environment variables only in development
if os.environ.get("FLASK_ENV") != "production":
    from dotenv import load_dotenv
    load_dotenv()

mongo_url = os.getenv('MONGO_URL')
client = pymongo.MongoClient(mongo_url)
db = client['test']
collection = db['batteries']
collection_forecasts = db['rangeforecasts']

data = pd.DataFrame(list(collection.find()))
existingData = pd.DataFrame(list(collection_forecasts.find()))


@app.route('/')
def home():
    print('Hello World')
    return 'Hello World'

@app.route('/predict/<userId>', methods=['POST'])
def GetData(userId):
    
    data['date'] = pd.to_datetime(data['date'], errors='coerce', utc=True)
    data['date'] = data['date'].dt.tz_localize(None)
    user = f'{userId}'
    
    user_data = data[data['userId'] == user]
    
    prophetData = pd.DataFrame({
        "ds": user_data['date'],
        'y': user_data['current_miles']
    })
    
    # Filter existingData for the same userId and the same date (ds)
    user_existing_data = existingData[(existingData['userId'] == user) & (existingData['ds'].isin(prophetData['ds']))]
    
    # Check if data already exists for the user on the same date or if there are fewer than 7 rows of data
    if not user_existing_data.empty or len(user_data) < 7:
        if not user_existing_data.empty:
            print(f"Data for user {user} on the same date has already been processed.")
            return 'Data has already been processed for this date and user', 200
        elif len(user_data) < 7:
            print(f"Not enough data for user {user}. Must have at least 7 rows of data.")
            return 'Not enough data to process', 400
    
    # Generate a new forecast as this is new data for this user
    model = Prophet()
    model.fit(prophetData)
    future = model.make_future_dataframe(periods=30, freq='D')
    forecast = model.predict(future)
    
    # Prepare the new data and insert it into MongoDB
    newData = pd.DataFrame({
        'ds': forecast['ds'],
        'yhat': forecast['yhat'],
        'userId': userId
    })
    
    newData = newData[newData['ds'] > pd.Timestamp.today()]   
    newData_dict = newData.to_dict(orient='records')  
    collection_forecasts.insert_many(newData_dict)

    return 'Data has been submitted to Database', 200
    
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
