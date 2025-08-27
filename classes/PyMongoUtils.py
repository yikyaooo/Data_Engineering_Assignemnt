from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime

class PyMongoUtils:
    def __init__(self, uri):
        self.client = MongoClient(uri, server_api=ServerApi('1'))
        
    def ping(self):
        try:
            self.client.admin.command('ping')
            print("Pinged your deployment. Successfully connected to MongoDB!")
        except Exception as e:
            print("Connection failed:", e)

    def get_database(self, db_name):
        db = self.client[db_name]
        return db

    def get_collection(self, db_name, collection_name):
        db = self.client[db_name]
        collection = db[collection_name]
        return collection

    def insert_single_data(self, collection, data):
        try:
            collection.insert_one(data)
            print("insert successfully!")
        except Exception as e:
            print("Error inserting:" , e)

    def insert_multiple_data(self, collection, df):
        try:
            records = [row.asDict() for row in df.collect()]
            collection.insert_many(records)
            print("insert successfully!")
        except Exception as e:
            print("Error inserting:" , e)
        
        