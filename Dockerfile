# Using Python 3.9 base image
FROM python:3.9

# Set the working directory to /code
WORKDIR /code

# Install necessary system dependencies, including libGL.so.1
RUN apt-get update && apt-get install -y libgl1 && apt-get clean

# Copy requirements.txt to /code
COPY ./requirements.txt /code/requirements.txt

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the entire project content to /code
COPY . /code

# CMD to run Gunicorn
CMD ["gunicorn", "main:app", "-b", "0.0.0.0:7860"]
