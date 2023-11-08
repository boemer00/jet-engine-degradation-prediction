# Start with a Python 3.10 base image
FROM python:3.10.6-slim

# Set the working directory in the container
WORKDIR /jet-engine

# Copy the current directory contents into the container
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run the application command when the container launches
CMD ["python", "./pipeline/rnn_pipeline.py"]
