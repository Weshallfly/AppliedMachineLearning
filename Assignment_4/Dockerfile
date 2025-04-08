# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# The /app directory should act as the main application directory
WORKDIR /app

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Copy all folder documents into the docker image; we will need this for testing
COPY . .

RUN pip install -r requirements.txt

# Expose the port on which the Flask app will run
EXPOSE 5000

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Command to run the Flask app
CMD ["python", "app.py"]
