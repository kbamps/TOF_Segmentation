# Use the base image from Docker Hub with the desired version of DeepVoxNet2.
FROM jeroenbertels/deepvoxnet2:latest

# Create directories to store application data inside the container.
RUN mkdir -p /app
RUN mkdir -p /in
RUN mkdir -p /out


# Install any required Python packages using pip.
RUN pip install pylibjpeg
RUN pip install opencv-python-headless
RUN pip install tqdm

# Copy the necessary files and folders into the container.
# Adjust 'main_folder' and 'predict_on_new_cases' accordingly to your project's directory structure.
ADD dicomorganizer /app/dicomorganizer
# ADD main_folder /app/main_folder
ADD main.py /app/main.py
ADD additional_functions /app/additional_functions

# Add app path to the Python path.
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Define the entry point for the Docker container.
# This is the command that will be executed when the container is sstarted.
# Modify the script name and arguments as needed for your application.
# In this case, the 'TOF_ACDC_predict_UZL_enhancedDCM.py' script will read input files from '/in'
# and save the output to '/out'.
ENTRYPOINT ["python", "/app/main.py"]
CMD ["--src_path", "/in", "--dst_path", "/out"]
