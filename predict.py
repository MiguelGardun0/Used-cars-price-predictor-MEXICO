from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Dict
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
import uvicorn

class Car(BaseModel):
    model_config = ConfigDict(extra="forbid")
    manufacturer: Literal[
        "acura", "alfa_romeo", "aston_martin", "audi", "bmw", "buick", "cadillac", 
        "chevrolet", "chrysler", "datsun", "dodge", "ferrari", "fiat", "ford", 
        "gmc", "harley_davidson", "honda", "hyundai", "infiniti", "jaguar", 
        "jeep", "kia", "land_rover", "lexus", "lincoln", "mazda", "mercedes_benz", 
        "mercury", "mini", "mitsubishi", "morgan", "nissan", "pontiac", "porsche", 
        "ram", "rover", "saturn", "subaru", "tesla", "toyota", "volkswagen", 
        "volvo", "unknown"
    ] = "unknown"
    model: str
    condition: Literal["excellent", "fair", "good", "like_new", "new", "salvage", "unknown"] = "unknown"
    cylinders: Literal[
        "10_cylinders", "12_cylinders", "3_cylinders", "4_cylinders", 
        "5_cylinders", "6_cylinders", "8_cylinders", "other", "unknown"
    ] = "unknown"
    fuel: Literal["diesel", "electric", "gas", "hybrid", "other", "unknown"] = "unknown"
    title_status: Literal["clean", "lien", "missing", "parts_only", "rebuilt", "salvage", "unknown"] = "unknown"
    transmission: Literal["automatic", "manual", "other", "unknown"] = "unknown"
    drive: Literal["4wd", "fwd", "rwd", "unknown"] = "unknown"
    size: Literal["compact", "full_size", "mid_size", "sub_compact", "unknown"] = "unknown"
    type: Literal[
        "bus", "convertible", "coupe", "hatchback", "mini_van", "offroad", 
        "other", "pickup", "sedan", "suv", "truck", "van", "wagon", "unknown"
    ] = "unknown"
    paint_color: Literal[
        "black", "blue", "brown", "custom", "green", "grey", "orange", 
        "purple", "red", "silver", "white", "yellow", "unknown"
    ] = "unknown"
    odometer: float = Field(..., ge=0,le=800000, description="Odometer reading in miles")
    year: int = Field(..., ge=1960, le=2026, description="Manufacturing year between 1960 and 2026")

class PredictResponse(BaseModel):
    predicted_price: float
    input_data : Dict

CAT_FEATURES = ["manufacturer", "model", "condition", "cylinders", "fuel", "title_status", "transmission", "drive", "size", "type", "paint_color"]
NUM_FEATURES = ["odometer", "year"]

loaded_model = joblib.load('model.pkl')
app = FastAPI(title="car_price")

@app.get("/")
def read_root():
    return {"car price": "predictor"}

@app.post("/prediction")
def predict(car: Car) -> PredictResponse:
    data = car.model_dump()
    X_new = pd.DataFrame([data])

    X_new["model"] = X_new["model"].str.strip().str.lower().str.replace(" ","_").str.replace("-","_")

    X_new = X_new[CAT_FEATURES + NUM_FEATURES] 

    prediction =loaded_model.predict(X_new)
    price = np.expm1(prediction[0])

    return PredictResponse (
        predicted_price = round(float(price),3),
        input_data = data
        )



if __name__ == "__main__":
    uvicorn.run("predict:app", host="127.0.0.1", port=8000, reload=True)

