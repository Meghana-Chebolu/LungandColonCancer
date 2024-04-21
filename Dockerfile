# Use an official TensorFlow runtime as a parent image
FROM python

# Set the working directory to /app
WORKDIR /app

# Copy requirements.txt to the /app directory
COPY requirements.txt /app/requirements.txt
COPY nrequirements.txt /app/nrequirements.txt


# Install system dependencies
RUN  apt-get update && \
     apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
     rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
RUN pip install -r nrequirements.txt

COPY . /app

# Make port 80 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
#uvicorn main:app --host 0.0.0.0 --port 8000 --reload
