from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Đường dẫn đến file dữ liệu bất động sản
DATA_FILE = 'du_lieu_bat_dong_san.csv'

# Hàm để huấn luyện mô hình
# Đọc dữ liệu, chia dữ liệu thành tập huấn luyện và tập kiểm tra, và huấn luyện mô hình hồi quy tuyến tính.
def train_model():
    df = pd.read_csv(DATA_FILE)
    X = df[['vi_tri', 'dien_tich', 'so_phong']]  # Các đặc trưng đầu vào
    y = df['gia']  # Giá bất động sản
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)  # Huấn luyện mô hình với dữ liệu huấn luyện
    return model

model = train_model()  # Lưu trữ mô hình đã huấn luyện

@app.route('/predict', methods=['POST'])
# Hàm để dự đoán giá bất động sản trên cơ sở dữ liệu đầu vào
# Nhận dữ liệu JSON từ yêu cầu POST và trả về giá dự đoán.
def predict():
    data = request.get_json()

    # Kiểm tra tính hợp lệ của dữ liệu đầu vào
    if 'vi_tri' not in data or 'dien_tich' not in data or 'so_phong' not in data:
        return jsonify({'error': 'Missing input data'}), 400

    vi_tri = data['vi_tri']
    dien_tich = data['dien_tich']
    so_phong = data['so_phong']

    # Chuyển đổi các đặc trưng thành dạng số trước khi dự đoán
    prediction = model.predict([[float(vi_tri), float(dien_tich), int(so_phong)]])  # Dự đoán giá
    return jsonify({'predicted_price': prediction[0]})  # Trả về giá dự đoán dưới dạng JSON

if __name__ == '__main__':
    app.run(debug=True)