import csv

import pandas as pd
import numpy as np

import argparse

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description="Generate \% of ranges of temperatures in the manjar flow")
parser.add_argument('-F', '--file', required=True, help='The complete path to the file')
parser.add_argument('-O', '--output', default='result.csv', help='The name of the output csv file')
parser.add_argument('--frio', default=24, help='minimum temperature')
parser.add_argument('--space', default=2, help='number of pixels in depth included in the average')
parser.add_argument('--span', default=2, help='number of pixels included in the average from the center')
parser.add_argument('--offset', default=3, help='offset of front included in the average')
parser.add_argument('--startY', default=100, help='Y coordinate of the top border')
parser.add_argument('--finishY', default=400, help='Y coordinate of the bottom border')
parser.add_argument('-S', '--startFrame', default=10, help='Frame to start')
parser.add_argument('-N', '--numberOfFrames', default=4000, help='Number of Frames of the file')
parser.add_argument('--interactive', default=False, help='should we graph it?')


args = parser.parse_args()
print(args)

start = int(args.startFrame)
frio = int(args.frio)

#lets make the graph interactive
if args.interactive:
    plt.ion()

data = []

def get_frame(reader):
    frame = []
    for _ in range(480):
        frame.append(next(reader))
    return pd.DataFrame(frame)

def processFrame(frame, space=3, span=7, offset=0, start=100, finish=400, interactive=True):
    frontY = []
    frontX = []
    for i in range(start, finish):
        for j, x in enumerate(frame.T[i]):
            if x > frio:
                frontX.append(j)
                frontY.append(i)
                break
        else:
            frontX.append(639)
            frontY.append(i)
            
    argminX = np.array(frontX).argmin()
    minY = frontY[argminX]
    minX = frontX[argminX]
    
    minY = int(minY + frame[minX][frame[minX] > frio].count() / 2)
    
    if interactive:
        plt.imshow(frame)
        #plt.scatter(frontX, frontY)
    
        plt.plot(np.linspace(0, 640, 640), np.repeat(minY, 640))
        plt.plot(np.repeat(minX, 480), np.linspace(0, 480, 480))
    
    area = []
    for i in range(minY - span, minY + span):
        for j, x in enumerate(frame.T[i]):
            if x > frio:
                area.append(frame.T[i][j + offset: j + offset + space].sum())
                if interactive:
                    plt.scatter([j + offset + k for k in range(space)], [i for k in range(space)])
                break
    if interactive:
        plt.pause(0.05)
        plt.clf()
    try:
        promedio = (sum(area) / len(area)) / space
    except:
        return 0, 0, 0
    return promedio, frontX, frontY


with open(args.file) as f:
    reader = csv.reader(f, delimiter=';')

    for x in range(int(args.numberOfFrames)):
        df = get_frame(reader)
        print('Frame {}'.format(x))
        if x < start:
            print('Skipping')
            continue
        df = df.applymap(lambda x: int(float(x.replace(',','.'))))
        
        promedio, frontx, fronty = processFrame(df, args.space, args.span, args.offset, args.startY, args.finishY, args.interactive)
        if promedio != 0:
            print('Frame added, the average was {}'.format(promedio))
            data.append(promedio)
        else:
            print('Frame skiped, problems getting the average')


newDF = pd.DataFrame(data)
newDF.to_csv(args.output)