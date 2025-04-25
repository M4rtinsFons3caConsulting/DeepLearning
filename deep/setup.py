""" 
Setup.py

This script as its name implies sets up the environment and data for the developer and tester.
It automatically fetches the data from the provided source, wrangles into `flow_from_dataframe`
compliance. 

Enriches the original metadata labels with binary images, adds hard negatives sourced from actual
image backgrounds,cleans the entire dataset using standard image treating protocol, and performs 
moderate oversampling using augmentation techniques.

"""

from deep.wrangle import drive_loader, resource_unpacker, directory_formatter

def main():
    print("Step 1: Loading source files from Google Drive")
    drive_loader.load_files()
    
    print("Step 2: Extracting supplementary archives")
    resource_unpacker.extract_archives()
    
    print("Step 3: Formatting dataset for flow_from_dataframe")
    directory_formatter.format_structure()
    
    print("Setup complete.")
    
if __name__ == "__main__":
   main()