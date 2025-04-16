# README

Getting started:

    1. Install the src package:

        To get started, create a conda environment with python version 3.11 or prior:
            
            - conda create <env_name> python==3.11

        Then install pip on the current env:
            - conta install pip

        Lastly, install the src package in the root directory:
            - pip install -e .

    2. Download the data:

        The data is available at <TK INSERT DRIVE link>, as it is part of the .gitignore once you have downloaded it
        migrate it to the data directory, and rename the entire directory to image_directory.

        Then run "_flatten_dir.py" as main.