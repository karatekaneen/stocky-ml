# FROM python:3.10-alpine
# WORKDIR /app
#
# ADD app /app/
# COPY model/ /app/model
#
# RUN pip install scikit-learn scipy matplotlib joblib flask pandas 
#
# CMD ["python", "app.py"]
#
# # Use the official Python image as the base imagedocker pull 
# FROM arm32v6/python:3.8-alpine
FROM python:3.9-slim
# FROM arm32v7/ython:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the application code to the container
COPY app/* .
COPY model/scaler /app/model/scaler

RUN pip install --no-cache-dir pandas numpy scikit-learn joblib flask
EXPOSE 5000
CMD ["python", "app.py"]
