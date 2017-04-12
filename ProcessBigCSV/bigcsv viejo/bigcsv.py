import csv

import pandas as pd
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Generate \% of ranges of temperatures in the manjar flow")
parser.add_argument('-F', '--file', required=True, help='The complete path to the file')
parser.add_argument('-O', '--output', default='result.csv', help='The name of the output csv file')
parser.add_argument('-LL', '--lowLimit', default=20, help='Low Limit to filter noise')
parser.add_argument('-HL', '--highLimit', default=40, help='High Limit to filter noise')
parser.add_argument('-MIN', '--minimum', default=30, help='Minimum value of temperature in manjar')
parser.add_argument('-MAX', '--maximum', default=65, help='Maximum value of temperature in manjar')
parser.add_argument('-R', '--numberOfRanges', default=4, help='Number of ranges to divide the spectrum')
parser.add_argument('-S', '--startFrame', default=10, help='Frame to start')
parser.add_argument('-N', '--numberOfFrames', default=4000, help='Number of Frames of the file')

args = parser.parse_args()

minimum = int(args.minimum)
maximum = int(args.maximum)
lowLimit = int(args.lowLimit)
highLimit = int(args.highLimit)
start = int(args.startFrame)

ranges = list(np.linspace(minimum, maximum, args.numberOfRanges))

data = []

def processFrame(df, enddata):
    frameData = []
    if df.mean().mean() < lowLimit:
        return
    elif df.mean().mean() > highLimit:
        return
    else:
        p100 = sum(df[df>=minimum].count())
        frameData.append(p100)
        for a, b in zip(ranges, ranges[1:]):
            frameData.append(sum(df[df>=a][df<b+0.001].count()) / p100)
        print(frameData)
        enddata.append(frameData)


def returnValues(temperature):
    n = 0.0046 * temperature + 0.1449
    k = -3.8484 * temperature + 255.16
    tau = -1.0362 * temperature + 84.246

    return (n, k, tau)


with open(args.file) as f:
    reader = csv.reader(f, delimiter=';')

    for x in range(int(args.numberOfFrames)):
        print('Frame {}'.format(x))
        frame = []
        for _ in range(480):
            frame.append(next(reader))
        if x < start:
            continue
        df = pd.DataFrame(frame)
        df = df.applymap(lambda x: int(float(x.replace(',','.'))))
        processFrame(df, data)

newDF = pd.DataFrame(data)
newDF.to_csv(args.output)
