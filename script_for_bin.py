#!/home/marcos/anaconda3/envs/numba/bin/python

import os
import bse_solver
import absorption

# ========================================================================= #
##                            MAIN FUNCTION
# ========================================================================= #
def main():
    bse_solver.main()
    absorption.main()
    # print(not os.path.isfile('infile_absorption.txt'))

if __name__ == '__main__':
    import timeit
    setup = "from __main__ import main"
    Ntimes = 1
    print(timeit.timeit("main()", setup=setup, number=Ntimes))
    # main()
