import pytest
from fastapi.testclient import TestClient
from app.predict import app 

client = TestClient(app)
@pytest.fixture
def car_data():
    return {
            "manufacturer": "mazda",
            "model": "mazda3",
            "condition": "like_new",
            "cylinders": "4_cylinders",
            "fuel": "gas",
            "title_status": "clean",
            "transmission": "automatic",
            "drive": "fwd",
            "size": "compact",
            "type": "hatchback",
            "paint_color": "black",
            "odometer": 5517,
            "year": 2025
            }



def test_predict_success(car_data):
    response = client.post("/prediction", json = car_data)

    assert response.status_code == 200
    assert "predicted_price" in response.json()

def test_predict_data_validation(car_data):
    car_data["year"] = "veinte-veinticinco"
    response = client.post("/prediction", json=car_data)

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "int_parsing"
    
def test_predict_extra_params(car_data):
    car_data["caballos"] = "198hp"
    response = client.post("/prediction", json=car_data)

    assert response.status_code == 422

        