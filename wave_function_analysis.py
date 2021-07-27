#!/home/marcos/anaconda3/envs/numba/bin/python
# import sys

import argparse
import numpy as np
import itertools as it

import treat_files as files
import hamiltonians as ham
import bethe_salpeter_equation as bse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--main_file", default="infile.txt",
                        type=str, help="path for the main input file")
    parser.add_argument("-b", "--bse_results", default="results_bse.npz",
                        type=str, help="path for the bse results file")
    args = parser.parse_args()
    main_input_file = args.main_file
    results_bse_file = args.bse_results
    return main_input_file, results_bse_file

def analyse_all_bse_states(A_Ns, Vectors, cond_n, vale_n, cond_vects, vale_vects):
    Ncomp = cond_n * vale_n
    nkx, nky,_,_ = Vectors.shape
    Nsites = nkx * nky
    Nbse_states = A_Ns.shape[1]
    All_bse_decomposed_array = np.empty((Nbse_states, Ncomp, Nsites), dtype=complex)
    for index in range(Nbse_states):
        All_bse_decomposed_array[index,:,:] = map_wavefunction_over_all_pairs(A_Ns[:,index], Vectors, cond_n, vale_n, cond_vects, vale_vects)
        # All_bse_decomposed_array.append(A_N_decomposed_list)
    # All_bse_decomposed_array = np.array(All_bse_decomposed_array)
    return All_bse_decomposed_array

def map_wavefunction_over_all_pairs(A_bse, Vectors, cond_n, vale_n, states_c, states_v):
    # A_list = []
    Ncomp = cond_n * vale_n ## NUMBER OF COMBINATIONS conduction-valence bands
    nkx, nky,_,_ = Vectors.shape
    Nsites = nkx * nky
    A_list = np.empty((Ncomp, Nsites), dtype=complex)
    index_comp = 0
    for v_state, c_state in it.product(states_v, states_c):
        A_proj = map_bse_wavefunction(A_bse, Vectors, cond_n, vale_n, c_state, v_state)
        # A_list.append(A_proj)
        A_list[index_comp,:] = A_proj
        index_comp += 1
    return A_list

def map_bse_wavefunction(A_bse, Vectors, cond_n, vale_n, state_c, state_v):
    ## VERIFY THE SHAPE OF THE INPUTS
    nkx, nky, nstates,_ = Vectors.shape # nkx -> kx-points; nky -> ky -> points; nstates -> # of eigenvalues
    Vectors_flat = Vectors.reshape(nkx*nky, nstates, nstates)
    N_combinations = cond_n * vale_n
    N_sites = nkx * nky

    ## BREAK THE BSE EIGENVECTOR IN TERMS OF Conduction-Valence PAIRS
    A_cv_list = break_A(A_bse, N_combinations, N_sites)
    c_states, v_states = break_V(Vectors_flat, cond_n, vale_n)

    ## CALCULATE THE PROJECTION OF THE STATE OVER |state_c, state_v>
    A_proj = np.sum([single_product(A, c_vectors, v_vectors, state_c, state_v)
               for A, (v_vectors, c_vectors) in zip(A_cv_list, it.product(v_states, c_states))],
               axis=0)
    return A_proj

def break_A(A_bse, Ncomb, Nsites):
    """
    INPUT  =  [A_c1v1k1, A_c2v1k1, ... , A_cNv1k1, A_c1v2k1, ..., A_cNvMkJ]
    OUTPUT = [[A_c1v1k1, A_c1v1k2, ... , A_c1v1kN],
              [A_c2v1k1, A_c2v1k2, ... , A_c2v1kN],
              :
              [A_cNvMk1, A_cNvMk2, ... , A_cNvMkN]]
    """
    decomp_list = np.empty((Ncomb, Nsites), dtype=complex)
    for n in range(Ncomb):
        # decomp_list.append(A_bse[n : : N_combinations])
        decomp_list[n,:] = (A_bse[n : : Ncomb])
    return decomp_list

def break_V(V_flat, cond_n, vale_n):
    cond_inds = list(range(-1,-1*(cond_n+1),-1)) # [-1,-2, ... , -cond_n]
    vale_inds = list(range(vale_n))              # [0,1,2, ... , (vale_n - 1)]
    c_states = [V_flat[:,:,c] for c in cond_inds]
    v_states = [V_flat[:,:,v] for v in vale_inds]
    return c_states, v_states

def single_product(A_cv, c_vectors, v_vectors, bra_cond, bra_vale):
    braket_cond = np.array([ *map(lambda c: bra_cond @ c, c_vectors) ])
    braket_vale = np.array([ *map(lambda v: bra_vale @ v, v_vectors) ])
    return A_cv * braket_cond * braket_vale

def main():
    # =============================== #
    ##      PARSE THE ARGUMENTS:
    # =============================== #
    main_input_file, results_bse_file = parse_arguments()
    results_bse = np.load(results_bse_file)
    print("Decomposing ", results_bse_file[:-4])

    # =============================== #
    ##      POP THE PARAMS OUT:
    # =============================== #
    params = files.read_params(main_input_file)
    Ham, r_0, epsilon, exchange, d_chosen, L_k, n_mesh, n_sub, submesh_radius, n_rec_states = files.pop_out_model(params)

    # =============================== #
    ##    DEFINE THE K-SPACE GRID:
    # =============================== #
    Kx, Ky, dk2 = bse.define_grid_k(L_k, n_mesh)
    grid = (Kx, Ky, dk2)

    # =============================== #
    ##    DEFINE THE HAMILTONIAN:
    # =============================== #
    hamiltonian = Ham(**params)
    cond_n, vale_n = hamiltonian.condBands, hamiltonian.valeBands
    cond_vects, vale_vects = hamiltonian.basis() ### THE HAMILTONIAN CLASS MAY NOT HAVE THIS METHOD DEFINED
    Values, Vectors = ham.values_and_vectors(hamiltonian, Kx, Ky) # Calculate the eigenvectors of H

    # =============================== #
    ##  LOAD THE RESULTS OF THE BSE
    # =============================== #
    all_bse_eigenvectors = results_bse['eigvecs_holder']
    all_bse_eigenvalues = results_bse['eigvals_holder']

    # =============================== #
    ##  CALCULATE THE PROJECTIONS
    # =============================== #
    All_bse_decomposed_array = analyse_all_bse_states(all_bse_eigenvectors, Vectors, cond_n, vale_n, cond_vects, vale_vects)

    # =============================== #
    ##  SAVE THE RESULTS
    # =============================== #
    output_name = results_bse_file[:-4] + "_decomposed.npz"

    print("Saving ", output_name)
    np.savez(output_name, eigenstates_decomposed=All_bse_decomposed_array,
                          eigenvalues=all_bse_eigenvalues)



if __name__=='__main__':
    # import timeit
    # setup = "from __main__ import main"
    # Ntimes = 1
    # print(timeit.timeit("main()", setup=setup, number=Ntimes))
    main()
