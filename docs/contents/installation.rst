.. docs/contents/installation.rst

Installation
============

To set up AIVUS-CAA on your system:

1. **Create a Python environment.** For example using `python3 -m venv` or conda (recommended).  

.. code-block:: bash

    python3 -m venv env
    source env/bin/activate
    pip install poetry
    poetry install


2. **Optional GPU setup (if using acceleration).** 
Install NVIDIA CUDA toolkit and drivers matching your TensorFlow version. For Ubuntu:

.. code-block:: bash
    sudo apt update && sudo apt upgrade
    sudo apt install build-essential dkms
    sudo ubuntu-drivers autoinstall
    sudo apt install nvidia-cuda-toolkit
    sudo reboot

- **Verify the NVIDIA driver**: 

.. code-block:: bash
    nvidia-smi

- **Test environment**: Ensure Python 3.10 or later is active and that all dependencies are installed.

3. **Launch AIVUS-CAA.** 
Run the main program:

.. code-block:: bash

    python3 src/main.py

The graphical user interface (GUI) should appear. If you encounter issues, review the README or submit an issue on GitHub.