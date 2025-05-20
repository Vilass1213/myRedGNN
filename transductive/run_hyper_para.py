#from trainer import *
import subprocess
import multiprocessing

def hidden_dim_task(i):
    sh = "python train.py --hidden_dim {:d}".format(i)
    # print(sh)
    subprocess.run(sh, shell=True)

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=2)

    hidden_dim = [8, 16]
    results = pool.map(hidden_dim_task, hidden_dim)