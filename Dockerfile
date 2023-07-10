FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Install system dependencies
RUN apt-get update && apt-get install -y nginx

# Create a work dir
WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY requirements_new.txt /app/requirements_new.txt

RUN pip install -r requirements.txt
RUN pip install -r requirements_new.txt

RUN python -c "import nltk;nltk.download('punkt')"
RUN python -c "import nltk;nltk.download('stopwords')"
RUN python -c "import nltk;nltk.download('averaged_perceptron_tagger')"

#COPY Models /app/Models
COPY app_enum.py /app/app_enum.py
COPY controller_.py /app/controller_.py
COPY data_cleaning.py /app/data_cleaning.py
COPY entities.py /app/entities.py
COPY helpers.py /app/helpers.py
COPY main.py /app/main.py
COPY model_enum.py /app/model_enum.py
COPY models.py /app/models.py
COPY build_downloader.py /app/build_downloader.py
RUN python build_downloader.py
# Set up Nginx
# Remove default Nginx configuration
RUN rm /etc/nginx/nginx.conf

# Copy custom Nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

ENV PYTHONUNBUFFERED 1

#RUN python build_downloader.py
#CMD ["uvicorn", "--workers", "1","main:app", "--host", "0.0.0.0", "--port", "8000"]
# Start Nginx and Uvicorn server
CMD service nginx start  && uvicorn main:app --host 0.0.0.0 --port 8000