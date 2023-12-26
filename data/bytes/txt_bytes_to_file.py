import argparse
import binascii

def parseargs():

  parser = argparse.ArgumentParser( description="Convert a file to a text representation of its bytes.")

  parser.add_argument('input_file', help='The input file containing the text representation')
  parser.add_argument('output_file', help='The output file to write back into original file format (e.g. mp3)')

  return parser.parse_args()

def text_to_file(input_file, output_file):
    # Read in text representation
    with open(input_file, 'r') as f:
        text_representation = f.read().rstrip('\n')
    original_content = binascii.unhexlify(text_representation.encode('utf-8'))

    # Export original format
    with open(output_file, 'wb') as f:
        f.write(original_content)

if __name__ == "__main__":
    args = parseargs()
    text_to_file(args.input_file, args.output_file)

