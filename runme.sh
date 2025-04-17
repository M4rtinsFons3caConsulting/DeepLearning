# Environment setup script (Linux Version)
#
# This script sets up the environment, and data in to the format and structure used by the team,
# it installs the source package to current environment using .toml file. Then it fetches the 
# /data from the Google Drive, flattens it, and appends the labels engineered by the group.
# 
# From then on, all of the notebooks should work without any issues, if you experience any issues
# in the process, please do not hesitate to contact us.

# Activate conda
conda activate # <directory name>

# Install src package to local directory
pip install -e . 

# Feching the data
<CODE TO GET THE DATA INTO /DATA/`IMAGE DIRECTORY` AND METADATA TO /DATA>

# Go to /src
cd src

# Setting up the directory structure
python dir_flattener.py

# Adding the initial binary labels
python label_merger.py --csv_file binary_labels --left_on rare_species_id --right_on image_id --how left

# Adding the cropped binary labels
python label_merger.py --csv_file cropped_labels --left_on rare_species_id --right_on image_id --how left