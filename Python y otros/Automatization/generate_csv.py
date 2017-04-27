
# coding: utf-8
import pandas as pd
import re
import os
import argparse

parser = argparse.ArgumentParser(description="Generate 1 file with all the data")
parser.add_argument('-F', '--folder', required=True, help='The complete path to the folder')
parser.add_argument('-O', '--output', default='result.csv', help='The name of the output csv file')
parser.add_argument('-R', '--regex', default=r'[A-Z]*(\d+)\.csv', help='Regex pattern for matching files')
args = parser.parse_args()
print(args.regex)
regex = re.compile(args.regex)

def generate_column(file):
    name = int(regex.search(file).group(1))
    return name, pd.read_csv(file, header=None, delimiter=';', usecols=[1], skiprows=2)

def apply_to_all(function, top):
    res = {}
    for root, paths, names in os.walk(top):
        print("Root: {}".format(root))
        if len(names) > 0:
            for name in names:
                print("Filepath: {}".format(os.path.join(root, name)))
                name, df = function(os.path.join(root, name))
                res[name] = df
    return res

dfs = pd.concat(apply_to_all(generate_column, args.folder), axis=1, ignore_index=True)
dfs.to_csv(args.output + '.csv', sep=';')
