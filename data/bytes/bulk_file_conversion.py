import argparse
import os
import binascii

def parseargs():
    parser = argparse.ArgumentParser(description='Convert all files in a folder to a single text file with byte representations.')

    parser.add_argument('folder_path', help='Path to the folder containing files to convert')
    parser.add_argument('output_file', help='The output file to write the text representations to')
    parser.add_argument('--delimiter', default='\n', help='Delimiter to separate files in the output (default: newline)')

    return parser.parse_args()

def file_to_text(input_file):
    with open(input_file, 'rb') as f:
        content = f.read()
    return binascii.hexlify(content).decode('utf-8')

def convert_folder_to_text(folder_path, output_file, delimiter='\n'):
    with open(output_file, 'w') as output:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                text_representation = file_to_text(file_path)
                output.write(text_representation + delimiter)

if __name__ == "__main__":
    args = parseargs()
    convert_folder_to_text(args.folder_path, args.output_file, args.delimiter)

