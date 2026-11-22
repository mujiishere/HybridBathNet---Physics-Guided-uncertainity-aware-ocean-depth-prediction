FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" rasterio pyproj shapely numpy pillow "psycopg[binary]" pydantic
COPY sdb_platform /app/sdb_platform
ENV PYTHONPATH=/app
EXPOSE 8000
