import argparse
import binascii
import os
import datetime

def parseargs():
    parser = argparse.ArgumentParser(description='Convert a text file with byte representations back into individual files.')

    parser.add_argument('input_file', help='The input file containing the byte representations')
    parser.add_argument('output_folder', help='The output folder to store the files')
    parser.add_argument('--file_extension', default='', help='File extension for the output files (default: no extension)')
    parser.add_argument('--discard_last', action='store_true', help='Discard everything after the last newline in the input file')
    parser.add_argument('--timestamp', action='store_true', help='Append a timestamp to the output folder name')

    return parser.parse_args()

def text_to_files(input_file, output_folder, file_extension='', discard_last=False):
    with open(input_file, 'r') as file:
        contents = file.read().rstrip('\n').split('\n') if discard_last else file.read().split('\n')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, content in enumerate(contents, start=1):
        file_name = f"{i}{file_extension}"
        file_path = os.path.join(output_folder, file_name)
        with open(file_path, 'wb') as output_file:
            output_file.write(binascii.unhexlify(content.encode('utf-8')))

if __name__ == "__main__":
    args = parseargs()

    if args.timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_folder = f"{args.output_folder}_{timestamp}"

    text_to_files(args.input_file, args.output_folder, args.file_extension, args.discard_last)

