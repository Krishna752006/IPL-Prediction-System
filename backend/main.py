from fastapi import FastAPI
from pydantic import BaseModel
from database import users_collection
from pydantic import EmailStr
import bcrypt
import re
from security import create_access_token, verify_token
from fastapi import HTTPException, Header, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from routers.prediction_history import router as history_router
from fastapi.middleware.cors import CORSMiddleware

from routers.predict import router as predict_router
from services.bowler_selector import router as bowler_router

app = FastAPI()
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(bowler_router)
app.include_router(history_router)


class RegisterUser(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str

class LoginUser(BaseModel):
    email: str
    password: str

def validate_password(password: str):

    if len(password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 8 characters."
        )

    if not re.search(r"[A-Z]", password):
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least one uppercase letter."
        )

    if not re.search(r"[a-z]", password):
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least one lowercase letter."
        )

    if not re.search(r"\d", password):
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least one number."
        )

    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least one special character."
        )

@app.get("/")
def home():
    return {"message": "Backend Running"}

@app.get("/me")
def me(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Token expired or invalid"
        )
    email = payload["sub"]
    user = users_collection.find_one(
        {"email": email},
        {"password": 0}
    )
    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    user["id"] = str(user["_id"])
    del user["_id"]
    return user

@app.post("/register")
def register(user: RegisterUser):

    if users_collection.find_one({"email": user.email}):
        return {"success": False, "message": "Email already registered"}

    validate_password(user.password)

    hashed_password = bcrypt.hashpw(
        user.password.encode(),
        bcrypt.gensalt()
    ).decode()

    users_collection.insert_one({
        "name": user.name,
        "email": user.email,
        "password": hashed_password,
        "role": user.role
    })

    return {
        "success": True,
        "message": "User registered successfully"
    }

@app.post("/login")
def login(user: LoginUser):

    db_user = users_collection.find_one({"email": user.email})

    if not db_user:
        return {
            "success": False,
            "message": "Invalid email or password"
        }

    password_matches = bcrypt.checkpw(
        user.password.encode(),
        db_user["password"].encode()
    )

    if not password_matches:
        return {
            "success": False,
            "message": "Invalid email or password"
        }

    token = create_access_token(
    {
        "sub": db_user["email"],
        "role": db_user["role"]
    }
)

    return {
        "success": True,
        "access_token": token,
        "token_type": "Bearer",
        "user": {
            "id": str(db_user["_id"]),
            "name": db_user["name"],
            "email": db_user["email"],
            "role": db_user["role"]
        }
    }
