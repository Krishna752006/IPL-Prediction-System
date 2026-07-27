from pymongo import MongoClient
from config import MONGO_URI

client = MongoClient(MONGO_URI)

db = client["ipl_prediction"]

users_collection = db["users"]
prediction_history_collection = db["prediction_history"]

print("Connected to MongoDB!")