import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_file', type=str)
parser.add_argument('--out_file', type=str)
parser.add_argument('--split', type=float)

args = parser.parse_args()

with open(args.in_file,"r") as in_file:
    lines = in_file.readlines()
keep = lines[:len(lines)*args.split]

with open(args.out_file,"w+") as out_file:
    out_file.writelines(keep)