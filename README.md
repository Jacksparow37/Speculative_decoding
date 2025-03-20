# Speculative_decoding
# Speculative Decoding over GPUs with Sockets

This repository contains a Python implementation of a speculative decoding algorithm using socket communication across multiple GPUs. The project includes two main components:

1. **Simple Socket Interaction**: A basic example (`socket_interaction.py`) demonstrating how to use sockets to communicate between processes running on different GPUs.
2. **Speculative Decoding Experiment**: A simplified speculative decoding implementation (`speculative_decoding.py`) where two draft models generate speculative tokens, and a primal model verifies them, producing final output sentences of approximately 30-40 words.

The code is designed to run on a system with at least 3 GPUs, using PyTorch for model inference and socket programming for inter-process communication.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Running the Code](#running-the-code)
  - [Test Socket Communication Over GPUs](#test-socket-communication-over-gpus)
  - [Run Speculative Decoding Experiment](#run-speculative-decoding-experiment)
- [Troubleshooting](#troubleshooting)
- [Additional Notes](#additional-notes)

## Prerequisites
Before running the code, ensure you have the following:

- **Hardware**:
  - A system with at least 3 GPUs (e.g., NVIDIA RTX 4090 with 24GB VRAM each).
  - CUDA-compatible GPUs with CUDA toolkit installed (e.g., CUDA 11.8 or compatible with your PyTorch version).

- **Software**:
  - Python 3.8 or higher.
  - PyTorch with CUDA support (e.g., `torch==2.0.1+cu118`).
  - Transformers library (`transformers==4.31.0`).
  - NumPy (`numpy==1.24.3`).

- **System Requirements**:
  - Linux or Windows with WSL2 (Windows Subsystem for Linux) for GPU support.
  - At least 24GB of VRAM per GPU to handle the GPT-2 models used in the experiment.

## Installation
Follow these steps to set up the environment and install the required dependencies.

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Jacksparrow37/speculative-decoding-sockets.git
   cd speculative-decoding-sockets
2. **Create the virtual environment**
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
3. **Install dependencies**
  ```bash
  pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
  pip install transformers==4.31.0 numpy==1.24.3
