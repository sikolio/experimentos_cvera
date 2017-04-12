import os
import csv
from ClaseExp import Experimento

Experimentos = []

with open('lista_exp.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        Experimentos.append(Experimento(row[0]))

#for exp in Experimentos:
#    print(exp.param)
#    print(exp.perfil)
#    exp.save_csv()
