# import os
import random

x = list(range(1,10000))
random.shuffle(x)

with open('seed_list.txt', 'w') as f:
    for n in x:
        f.write(str(n)+'\n')