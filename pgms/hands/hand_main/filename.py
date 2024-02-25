import os
import glob

def filen(folder_path):
    # Define the folder path
    #folder_path = r'E:\website files\bodylang\static\myapp\videos\person1\face'

    # Get a list of all files in the folder
    files = glob.glob(os.path.join(folder_path, '*'))

    # Sort the files by modification time to find the last uploaded file
    files.sort(key=os.path.getmtime, reverse=True)

    # Check if any files were found in the folder
    if files:
        # Get the path of the last uploaded file
        last_uploaded_file = files[0]
        
        # Extract the file name without extension
        file_name, file_extension = os.path.splitext(os.path.basename(last_uploaded_file))
        
        # Check if the file name ends with a number
        if file_name[-1].isdigit():
            # Extract the number from the file name and increment it
            num = int(''.join(filter(str.isdigit, file_name)))
            num += 1
            # Create the new output file path with the incremented number
            output_file_path = os.path.join(folder_path, f"{file_name[:-1]}{num}{file_extension}")
        else:
            # If the last uploaded file doesn't end with a number, append "1" to the filename
            output_file_path = os.path.join(folder_path, f"{file_name}1{file_extension}")
    else:
        # If no files were found in the folder, set a default output path
        output_file_path = os.path.join(folder_path, 'output_video.mp4')

    return output_file_path
