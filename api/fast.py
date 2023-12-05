# TODO: Import your package, replace this by explicit imports of what you need

from api.fetch_btc_data import get_btc_data
from financial_app.preprocessor import *
from financial_app.utils import *
from financial_app.data import features_engineering
from financial_app.registry import load_model


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {
        'message': "Hi, The API is running!"
    }

# Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
@app.get("/predict")
def get_predict(model_name='BASELINE'):

    if model_name=='BASELINE':
        model = load_model()

    btc_data = get_btc_data()

    X_test = btc_data

    X_test = features_engineering(X_test)

    X_test_processed = preprocessor(X_test).to_numpy()

    X_test_processed = X_test_processed[-15:,]

    X_test_processed = np.expand_dims(X_test_processed, axis=0)

    prediction = model.predict(X_test_processed)

    response = {
        'prediction': float(prediction[0][0])
    }

    return response

if __name__ == "__main__":


    print(get_predict("BASELINE"))
