import os 
import sys

first=int(sys.argv[1])
second=int(sys.argv[2])

jobs=[first + f for f in range(second+1-first)]

for j, job in enumerate(jobs):
	os.system("scancel " + str(job))

