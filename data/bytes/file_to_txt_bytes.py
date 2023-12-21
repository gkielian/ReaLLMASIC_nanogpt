import argparse
import binascii

def parseargs():

  parser = argparse.ArgumentParser( description="Convert a file to a text representation of its bytes.")

  parser.add_argument("input_file", help="Input file to be converted to bytes")
  parser.add_argument("output_file", help="The output file to write the text representation of bytes to")

  return parser.parse_args()

def file_to_text(input_file, output_file):
    # Read in content from input file
    with open(input_file, "rb") as f:
        content = f.read()

    text_representation = binascii.hexlify(content).decode("utf-8")

    # Write results to output file
    with open(output_file, "w") as f:
        f.write(text_representation)

if __name__ == "__main__":
    args = parseargs()
    file_to_text(args.input_file, args.output_file)
