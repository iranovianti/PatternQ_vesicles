FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY patternq_cli.py /app/
COPY dataset.py /app/
COPY model_defs.py /app/

ENTRYPOINT ["python", "patternq_cli.py"]