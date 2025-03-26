# Use an appropriate base image
FROM python:3.11-slim

# Set the working directory
WORKDIR .

# Copy the requirements file and install dependencies
COPY requirements.txt .
COPY input_analysis.py .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port on which the Dash app runs
EXPOSE 8050

# Command to run the application
CMD ["python", "input_analysis.py"]
