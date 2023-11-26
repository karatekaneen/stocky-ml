FROM python:3.11-alpine
WORKDIR /app

ADD ./app /app
ADD ./model /app/model

RUN pip install scikit-learn scipy matplotlib joblib flask pandas

CMD ["python","app.py"]