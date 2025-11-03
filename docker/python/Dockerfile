FROM python:3.13-slim

WORKDIR /python

# Установка системных библиотек
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    python3-dev \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

COPY ./docker/python .

COPY ./requirements.txt .

# Установка зависимостей Python
RUN pip install --upgrade pip
RUN pip install -v --no-cache-dir -r ./requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]