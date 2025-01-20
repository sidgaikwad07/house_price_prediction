# Use the official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libpng-dev \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python libraries
RUN pip install --upgrade pip \
    && pip install jupyter notebook \
    numpy pandas matplotlib seaborn \
    scikit-learn==1.6.1 scipy \
    xgboost==2.1.3 lightgbm==4.0.0 catboost \
    statsmodels \
    plotly \
    ipywidgets \
    jupyterlab \
    requests beautifulsoup4 \
    opencv-python \
    nltk spacy \
    pyarrow fastparquet \
    pyyaml \
    joblib

# Download SpaCy language model
RUN python -m spacy download en_core_web_sm

# Expose the default Jupyter port
EXPOSE 8888

# Run Jupyter Notebook without authentication
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

