import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_file', type=str)
parser.add_argument('--out_file', type=str)
parser.add_argument('--split', type=int)

args = parser.parse_args()

with open(args.in_file) as in_file:
    lines = in_file.readlines()
keep = lines[:len(lines)*100//args.split]

with open(args.out_file) as out_file:
    out_file.writelines(keep)