"""Connection helper for the PostGIS database defined in docker-compose.yml."""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


def get_engine():
    url = os.environ.get("DATABASE_URL", "postgresql://ndvi:ndvi@localhost:5432/ndvi")
    return create_engine(url)
