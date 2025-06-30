from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
from urllib.parse import quote_plus

class Database:
    def __init__(self):
        uri = "mongodb+srv://maanyaa23:9T4BuYB4BZKvr5LP@cluster0.yubbqgj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        # Create a new client and connect to the server
        client = MongoClient(uri, server_api=ServerApi('1'))
        # Send a ping to confirm a successful connection
        try:
            client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)

        # Connect to the database
        db = client.get_database('walmart_sales')

        # Connect to the collections
        self.records = db['SalesPredictor']

    def add_single_document(self, input_data):
        try:
            result = self.records.insert_one(input_data)
            return result
        except Exception as e:
            print("Insert failed")

    def delete_all_documents(self):
        try:
            self.records.delete_many({})
        except Exception as e:
            print('Deletion Failed')

if __name__ == '__main__':
    D = Database()
    D.delete_all_documents()
