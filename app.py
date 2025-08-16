from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Đường dẫn đến file dữ liệu bất động sản
DATA_FILE = 'du_lieu_bat_dong_san.csv'

def train_model():
    df = pd.read_csv(DATA_FILE)
    X = df[['vi_tri', 'dien_tich', 'so_phong']]
    y = df['gia']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    vi_tri = data['vi_tri']
    dien_tich = data['dien_tich']
    so_phong = data['so_phong']
    prediction = model.predict([[vi_tri, dien_tich, so_phong]])
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
