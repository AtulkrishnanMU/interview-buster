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

        # Initialize a variable to store the extracted number
        extracted_number = ''

        # Iterate through the characters in the file name
        for char in reversed(file_name):
            if char.isdigit():
                # If the character is a digit, prepend it to the extracted_number
                extracted_number = char + extracted_number
            else:
                # If a non-digit character is encountered, break the loop
                break

        if extracted_number:
            # Convert the extracted number to an integer
            num = int(extracted_number)
            # Increment the number
            num += 1
            # Create the new output file path with the incremented number
            output_file_path = os.path.join(folder_path, f"{file_name[:-len(extracted_number)]}{num}{file_extension}")
        else:
            # If no number was extracted, append "1" to the filename
            output_file_path = os.path.join(folder_path, f"{file_name}1{file_extension}")
    else:
        # If no files were found in the folder, set a default output path
        output_file_path = os.path.join(folder_path, 'output.jpg')

    return output_file_path
