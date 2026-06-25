# =============================================================================
#  web/Dockerfile  —  Flask + Gemini analyze server
#   - Do NOT bake the API key into the image. Inject at runtime via --env-file.
#       docker build -t beehive-web .
#       docker run --env-file ../.env -p 5000:5000 beehive-web
#   - PYTHONUNBUFFERED so logs (Flask banner, tracebacks) stream immediately;
#     without it the container can look "silent" even though it is running.
# =============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "-u", "app.py"]
