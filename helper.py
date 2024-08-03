#for line in txt filename, use regular expression if it has "ETA", delete line
import re 

def remove_ETA(file):
    with open(file, "r") as f:
        lines = f.readlines()
    with open(file, "w") as f:
        for line in lines:
            if not re.search("ETA", line):
                f.write(line)
#applie to file called slurm.txt
remove_ETA("slurm-25352384.txt")