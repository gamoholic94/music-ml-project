# This first command sets the foundation for our entire build process.
# We are telling Docker to start with the official Python image.
# The ":3.9-slim" tag specifies that we want a minimal version
# of the image that includes Python version 3.9. This keeps our
# final application size smaller and more efficient.
FROM python:3.9-slim

# The WORKDIR instruction sets the working directory for any subsequent
# RUN, CMD, COPY, or ADD instructions in this Dockerfile.
# This is a best practice that helps keep the container's filesystem tidy.
# All our application files will live inside /app.
WORKDIR /app

# The COPY instruction transfers files from the build context (your project
# directory on your local machine) into the container's filesystem.

# First, copy the requirements file. We do this as a separate step
# to take advantage of Docker's layer caching.
COPY requirements.txt .

# Now, copy the rest of the essential application files.
COPY app.py .
COPY music_genre_cnn.h5 .
COPY scaler.joblib .

# The RUN instruction executes commands inside the image during the build process.
# Here, we are installing all the Python dependencies listed in requirements.txt.
# We first upgrade pip to ensure we have the latest version.
# The `--no-cache-dir` flag is a crucial optimization that prevents pip from
# storing the downloaded package files, keeping our final image size smaller.
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# The EXPOSE instruction informs Docker that the container will listen on the
# specified network port at runtime. Streamlit's default port is 8501.
# This is a piece of documentation for the image.
EXPOSE 8501

# The CMD instruction specifies the default command to run when a container
# is started from this image. We use the "exec" form, which is the best practice.
# This command starts the Streamlit server and runs our app.py script.
CMD ["streamlit", "run", "app.py"]