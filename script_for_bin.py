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

if __name__ == '__main__':
    main()
