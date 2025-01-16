from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd

app = FastAPI()

# Enable CORS for all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os 
print("Files in directory:", os.listdir())
# Load the trained CatBoost model
with open('best_catboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

class InputData(BaseModel):
    temperature: float
    co: float
    no2: float
    humidity: float
    pop_density: float

@app.get("/")
def home():
    return {"message": "Welcome to the Air Quality Prediction API"}

@app.post("/predict")
def predict(data: InputData):
    try:
        temp = data.temperature
        co = data.co
        no2 = data.no2
        humidity = data.humidity
        pop_density = data.pop_density
        
        # Calculate interaction terms
        temp_co = temp * co
        no2_co = no2 * co
        humidity_co = humidity * co
        co_pop_density = co * pop_density

        input_data = pd.DataFrame({
            'CO': [co],
            'Temperature CO': [temp_co],
            'NO2 CO': [no2_co],
            'Humidity CO': [humidity_co],
            'CO Population_Density': [co_pop_density]
        })
        
        prediction = model.predict(input_data)[0]
        result = "Good" if prediction == 1.0 else "Poor"
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
