# TODO: Import your package, replace this by explicit imports of what you need

from api.fetch_btc_data import get_btc_data
from api.fetch_stock_data import get_stock_data
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


time_horizon_mapping = {
    '5mn': '5m',
    '5m': '5mn',
    '1h': '60m',
    '60m': '1h'
}


# Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
@app.get("/predict")
def get_predict(asset='btc-usd', time_horizon='1h', model_type='gru'):

    # Convert the time horizon using the mapping
    time_horizon = time_horizon_mapping.get(time_horizon, time_horizon)

    real_time_price = get_stock_data(asset, time_horizon)

    # Convert the time horizon back
    time_horizon = time_horizon_mapping.get(time_horizon, time_horizon)


    if asset == 'btc-usd':
        asset = 'btc'

    model_name = f'{asset}_{model_type}_{time_horizon}'

    model = load_model(model_name)

    X_test = real_time_price

    X_test = features_engineering(X_test)

    X_test_processed = preprocessor(X_test).to_numpy()

    X_test_processed = X_test_processed[-10:,]

    X_test_processed = np.expand_dims(X_test_processed, axis=0)

    prediction = model.predict(X_test_processed)

    response = {
        'prediction': float(prediction[0][0])
    }

    return response

@app.get("/hist_predict")
def get_hist_predict(asset='btc-usd', time_horizon='1h', model_type='gru'):

    # Convert the time horizon using the mapping
    time_horizon = time_horizon_mapping.get(time_horizon, time_horizon)

    real_time_price = get_stock_data(asset, time_horizon)

    # Convert the time horizon back
    time_horizon = time_horizon_mapping.get(time_horizon, time_horizon)

    if asset == 'btc-usd':
        asset = 'btc'

    model_name = f'{asset}_{model_type}_{time_horizon}'

    model = load_model(model_name)

    X_test = real_time_price

    y_test = labeling_df(X_test)
    y_test = y_test['label'].shift(-1)
    X_test = X_test.drop(columns=['label'])
    X_test = features_engineering(X_test)


    X_test_processed = preprocessor(X_test).to_numpy()


    prediction_list = []

    # Loop over each element considering batches of 10 at a time
    for k in range(len(X_test_processed)):
        X_test_to_predict = X_test_processed[k:k+10,]
        X_test_to_predict = np.expand_dims(X_test_to_predict, axis=0)
        prediction = model.predict(X_test_to_predict)[0][0]
        prediction_list.append(prediction)


    pred_label = [0 if prediction < 0.5 else 1 for prediction in prediction_list ]

    predictions_df = pd.DataFrame(pred_label, columns=['pred_label'])

    # Concatenate y_test and predictions_df side by side
    ct_df = pd.concat([y_test[-len(prediction_list):].reset_index(drop=True),
                       X_test['adj_close'][-len(prediction_list):].reset_index(drop=True),
                       predictions_df], axis=1).dropna()


    response = {
        'label': list(ct_df['label']),
        'close': list(ct_df['adj_close']),
        'pred_label': list(ct_df['pred_label'])
    }


    return response



if __name__ == "__main__":

     print(get_hist_predict("btc-usd", "5mn", "gru"))
