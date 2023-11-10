# Start with a Python 3.10 base image
FROM python:3.10.6-slim

# Set the working directory in the container
WORKDIR /jet-engine

# Copy the current directory contents into the container
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Uvicorn for serving the application
RUN pip install uvicorn

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the Uvicorn server when the container launches
CMD ["uvicorn", "src.app.app:app", "--host", "0.0.0.0", "--port", "8000"]
