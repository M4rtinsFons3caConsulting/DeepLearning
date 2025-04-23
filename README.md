# README

# Project – Setup and Installation

This folder contains setup and installation guidelines for getting started with the project.  
> **Note:** These instructions assume a clean environment and are intended to set up everything needed for the project.

## Included Steps

- **`pip install -e .`**  
  Install the project package in editable mode.

- **`setup`**  
  Initializes necessary directories, augments the original data, and generates the appropriate labels.

---

These steps were designed to set up the initial environment for the project.  
They are **meant to be run at the beginning of the project setup**.

For more advanced or customized workflows, you may want to modify the scripts based on your environment.

## Usage

Each step can be executed via the command line. To display help or pass runtime arguments, follow this general pattern:

```bash
# For Linux
cd ../project/
python setup --help

# For Windows
cd ..\project\
python setup --help
```
