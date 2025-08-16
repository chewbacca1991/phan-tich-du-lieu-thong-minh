import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class RealEstateModel:
    def __init__(self, data_file):
        self.data_file = data_file
        self.model = None

    def train(self):
        df = pd.read_csv(self.data_file)
        X = df[['vi_tri', 'dien_tich', 'so_phong']]
        y = df['gia']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, input_data):
        return self.model.predict([input_data])
