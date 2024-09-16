import random
import argparse
parser = argparse.ArgumentParser(description='Process a file and create weights.')
parser.add_argument('input_file', type=str, help='Input file name')
args = parser.parse_args()

FILE = args.input_file

def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if not line.startswith("#"):
                line = line.strip() + "\t" + str(random.randint(1, 100)) + "\n"
            outfile.write(line)

process_file(FILE, f"weights_{FILE}")