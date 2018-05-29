import numpy as np
import math


with open('svmguide1', 'r') as f:
    X = f.read()
    X = X.split('\n')
    indices = []
    numbers = []
    for i in range(len(X)):
        index = []
        number = []
        X[i] = X[i].split(' ')
        for j in range(1, len(X[i])):
            tmp = X[i][j].split(':')
            index.append(tmp[0])
            number.append(math.sqrt(abs(float(tmp[1]))))
        indices.append(index)
        numbers.append(number)
    f.close()

with open('svmguide1.suqare', 'w') as f:
    for i in range(len(X)):
        output = X[i][0]
        for j in range(len(indices[i])):
            output += ' %s:%f'%(indices[i][j], numbers[i][j])
        print(output, file=f)
    f.close()


            
