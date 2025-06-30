import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDRegressor  # Stochastic Gradient Descent
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

class Model:
    def __init__(self):
        uri = "mongodb+srv://maanyaa23:9T4BuYB4BZKvr5LP@cluster0.yubbqgj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        client = MongoClient(uri, server_api=ServerApi('1'))
        db = client.get_database('walmart_sales')
        self.records = db['SalesPredictor']
        self.df = pd.DataFrame(list(self.records.find({})))

    def populate_db(self, filepath):
        # First clear existing documents
        self.records.delete_many({})  # WARNING: This removes all existing data

        # Load new data
        df = pd.read_csv(filepath)
        examples = df.to_dict(orient='records')

        # Insert fresh records
        self.records.insert_many(examples)
        print("Database repopulated with", len(examples), "records.")

    def train(self):
        self.df.drop(columns=['_id'], inplace=True, errors='ignore')
        print(self.df.head())
        X = self.df.drop(['Store','Date','Weekly_Sales'], axis=1)
        y = self.df['Weekly_Sales']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, 'pipeline_model.pkl')
        loaded_pipeline = joblib.load('pipeline_model.pkl')
        y_test_pred = loaded_pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        return r2

if __name__ == '__main__':
    m = Model()
    m.populate_db('Walmart_Sales.csv')
    s = m.train()
    print("R2 Score: ", str(s))
