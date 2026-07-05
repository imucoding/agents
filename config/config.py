from pathlib import Path
from dotenv import dotenv_values, load_dotenv
import os

class Config:
    def __init__(self):
        root_path = Path(__file__).resolve().parents[1]
        env_path = root_path / ".env"
        load_dotenv(dotenv_path=env_path)