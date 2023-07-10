FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Create a work dir
WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY requirements_new.txt /app/requirements_new.txt

RUN pip install -r requirements.txt
RUN pip install -r requirements_new.txt

#RUN python -c "import nltk;nltk.download('punkt')"
#RUN python -c "import nltk;nltk.download('stopwords')"
#RUN python -c "import nltk;nltk.download('averaged_perceptron_tagger')"

COPY Models /app/Models
COPY app_enum.py /app/app_enum.py
COPY controller_.py /app/controller_.py
COPY data_cleaning.py /app/data_cleaning.py
COPY entities.py /app/entities.py
COPY helpers.py /app/helpers.py
COPY main.py /app/main.py
COPY model_enum.py /app/model_enum.py
COPY models.py /app/models.py

ENV PYTHONUNBUFFERED 1

CMD ["uvicorn", "--workers", "1","main:app", "--host", "0.0.0.0", "--port", "8000"]
