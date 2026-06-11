import joblib
import pandas as pd

# 1. Load the saved assets
loaded_model = joblib.load('models/trading_model.joblib')
loaded_le = joblib.load('models/label_encoder.joblib')

# 2. Define a future scenario
future_side = 'BUY'
future_coin = 'BTC'
future_sentiment_val = 15 # Extreme Fear

# 3. Preprocess the input (Map classification just like training)
if future_sentiment_val <= 25: future_class = 'Extreme Fear'
elif future_sentiment_val <= 45: future_class = 'Fear'
elif future_sentiment_val <= 55: future_class = 'Neutral'
elif future_sentiment_val <= 75: future_class = 'Greed'
else: future_class = 'Extreme Greed'

# 4. Prepare DataFrame and encode
input_df = pd.DataFrame([{
    'Side': future_side,
    'value': future_sentiment_val,
    'classification': future_class,
    'Coin': future_coin
}])

for col in ['Side', 'classification', 'Coin']:
    input_df[col] = loaded_le.fit_transform(input_df[col].astype(str))

# 5. Predict
prediction = loaded_model.predict(input_df)[0]
prob = loaded_model.predict_proba(input_df)[0][1]

print(f"Loaded Model Prediction: {'PROFITABLE' if prediction == 1 else 'NOT PROFITABLE'}")
print(f"Winning Probability: {prob:.2%}")