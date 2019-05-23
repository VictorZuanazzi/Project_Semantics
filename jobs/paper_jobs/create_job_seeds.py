from glob import glob 
import numpy as np 
import os 
import sys

INITIAL_JOB_FILE = "Hierarchy_MNLI_VUAseq_POS_1.job"
NUM_SEEDS = 16
START_SEED = 42

with open(INITIAL_JOB_FILE, "r") as f:
	default_file = "".join(f.readlines())

for n in range(NUM_SEEDS):
	seed_file = default_file.replace("seed_"+str(START_SEED), "seed_" + str(START_SEED+n)).replace("seed "+str(START_SEED), "seed "+str(START_SEED+n))
	seed_filename = INITIAL_JOB_FILE.rsplit("_",1)[0] + "_" + str(n+1) + ".job"
	with open(seed_filename, "w") as f:
		f.write(seed_file)
