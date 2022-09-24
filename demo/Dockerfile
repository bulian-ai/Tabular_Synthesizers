# app/Dockerfile

FROM python:3.7-slim

EXPOSE 8501

WORKDIR /demo

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install git+https://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@github.com/bulian-ai/Tabular_Synthesizers.git

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]