import os

from pymongo import MongoClient

mongo_uri = os.getenv("MONGO_URI")

client = MongoClient(mongo_uri)
db = client["ipl_db"]


def check_db():
    try:
        client.admin.command("ping")
        return {"db": "connected"}
    except Exception as e:
        return {"db": "error", "message": str(e)}


@app.get("/db-check")
def db_check():
    return check_db()
