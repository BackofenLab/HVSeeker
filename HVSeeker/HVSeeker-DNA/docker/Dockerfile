# Use CUDA base image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set working directory
#WORKDIR /mnt/docker2/HVSeeker/HVSeeker/HVSeeker-DNA/
WORKDIR /app
# Install dependencies for Python, pip, and wget
RUN apt-get update && apt-get install -y python3.10 python3-pip wget bzip2 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add Conda to the PATH
ENV PATH=/opt/conda/bin:$PATH

# Create and activate Conda environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate the environment
SHELL ["/bin/bash", "--login", "-c"]
RUN echo "source activate HVSeekerDNA" >> ~/.bashrc

# Install PyTorch within the Conda environment using pip
RUN conda run -n HVSeekerDNA pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy your Python scripts
COPY DataParsing_main.py .
COPY DNA_Predictor.py .
COPY DNA_Dataset_creator_txt.py .
COPY make_datasets.py .
COPY DNA_Trainer.py .
COPY DNA_Prediction_Preprocessing.py .
COPY DNA_Data_Preprocessing.py .
COPY main.py .

# Run the main script with the Conda environment
#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "HVSeekerDNA", "python", "main.py"]

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "HVSeekerDNA"]
CMD ["python -u"]
