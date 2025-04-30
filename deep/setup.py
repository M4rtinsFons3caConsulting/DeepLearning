"""
setup.py

This script sets up the environment and prepares the dataset for development and testing.
It automatically fetches the data from the provided sources and wrangles it into 
`flow_from_dataframe` compliance.

The script enriches the original metadata with binary labels for animal presence, 
adds hard negatives sourced from background images, cleans the dataset using standard 
image preprocessing protocols, and performs moderate oversampling through augmentation techniques.
"""


from deep.wrangle import drive_loader, directory_maker, resource_unpacker, directory_formatter 

def main():
    print("Step 1: Loading source files from Google Drive")
    drive_loader.load_files()
    
    print("Step 2: Making the necessary directories")
    directory_maker.make_dir()

    print("Step 3: Extracting supplementary archives")
    resource_unpacker.extract_archives()
    
    print("Step 4: Formatting dataset for flow_from_dataframe")
    directory_formatter.format_structure()
    
    print("Setup complete.")
    
if __name__ == "__main__":
   main()