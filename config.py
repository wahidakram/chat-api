import os
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


class ModeEnum(str, Enum):
    development = "development"
    production = "production"
    testing = "testing"


class Config:
    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = os.getenv("REDIS_PORT", 6379)
    REDIS_DB = os.getenv("REDIS_DB", 0)
    REDIS_DECODE_RESPONSES = os.getenv("REDIS_DECODE_RESPONSES", True)

    FAISS_DB_FILE = os.getenv("FAISS_DB_FILE", "")
    FAISS_DB_FILE_TEMP = os.getenv("FAISS_DB_FILE_TEMP", "")
    SERVER_URL = os.getenv("SERVER_URL", "")


config = Config()
