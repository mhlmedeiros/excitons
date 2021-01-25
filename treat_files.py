
import numpy as np
import os, shutil, subprocess
import hamiltonians as ham

# ========================================================================= #
##                            'READ INPUT FILE
# ========================================================================= #
def open_template(specific):
    path_src = '/home/marcos/Documents/DAAD_Research/excitons/excitons_python'
    template = 'template_' + specific
    path_dst = os.getcwd()
    file_template = path_src + '/' + template
    file_specific = path_dst + '/' + specific
    shutil.copy(file_template, file_specific)
    subprocess.call(['vim', specific])
    return 0

def open_infile(specific):
    path_dst = os.getcwd()
    file_specific = path_dst + '/' + specific
    subprocess.call(['vim', specific])
    return 0

def read_file(specific):
    current_path = os.getcwd()
    file  = open(specific,'r');
    lines = file.readlines()
    file.close()
    params = {}
    for line in lines:
        list_line = line.split()
        if len(list_line) == 0 or list_line[0]=='##': continue
        else: params[list_line[0]] = list_line[2]
    return params

def verify_essential_file(name_file):
    has_input  = os.path.isfile(name_file)
    if not has_input :
        print("Please, run 'exciton -bse' to generate the eingenvectors.")
        exit()

def verify_file_or_template(name_file):
    has_input   = os.path.isfile(name_file)
    if not has_input  : open_template(name_file)
    else              : open_infile(name_file)

def from_dic_to_var(Ham, gamma, Egap, r0, mc, mv, alpha_option, epsilon, L_k, n_mesh, n_sub, submesh_radius, n_rec_states):

    ## MODEL HAMILTONIAN
    Ham           = eval('ham.' + Ham)  # chosen Hamiltonian has to be a class in 'hamiltonians.py'
    gamma         = float(gamma)        # meV*nm
    Egap          = float(Egap)         # meV
    r0            = float(r0)           # nm
    mc            = float(mc)           # conduction band effective mass
    mv            = float(mv)           # valence band effective mass
    alpha_option  = str(alpha_option)   # options: (zero, masses, corrected)
    epsilon       = float(epsilon)      # effective dielectric constant

    ## K-SPACE
    L_k             = float(L_k)                # the discretized k-space limits: k_x(y) ∈ [-L,L]
    n_mesh          = int(n_mesh)               # quantity of points in each direction
    n_sub           = int(n_sub)                # quantity of points in each of the submesh direction
    submesh_radius  = int(submesh_radius)       # in number of sites; radius of the region where the submesh will be considered

    ## SAVING OPTIONS:
    n_rec_states    = int(n_rec_states)         # number of states to be recorded

    return Ham, gamma, Egap, r0, mc, mv, alpha_option, epsilon, L_k, n_mesh, n_sub, submesh_radius, n_rec_states

def from_dic_to_var_abs(pol_option, p_matrix, Gamma_option, Gamma1, Gamma2, Gamma3, padding, N_points_broad):
    ## LIGHT POLARIZATION:
    pol_option  = str(pol_option)       # options = {x, y, c_plus, c_minus}
    p_matrix    = str(p_matrix)         # options = {P_2x2, P_4x4}

    ## CONVOLUTION:
    Gamma_option    =  str(Gamma_option)        # options = {C = constant: Gamma1; V = Γ(ħω; Γ_1, Γ_2, Γ_3)}
    Gamma1          = float(Gamma1)             # Γ_1 = Width at half-maximum (used in options: C and V)
    Gamma2          = float(Gamma2)             # Γ_2 = Width at half-maximum (used in option: V)
    Gamma3          = float(Gamma3)             # Γ_3 = Width at half-maximum (used in option: V)
    padding         = float(padding)            # meV padding
    N_points_broad  = int(N_points_broad  )   # E_broad = np.linspace(E_raw[0]-padding, E_raw[-1]+padding, N_points_broad)
    return pol_option, p_matrix, Gamma_option, Gamma1, Gamma2, Gamma3, padding, N_points_broad

# ========================================================================= #
##                         WRITING THE OUTPUT FILE
# ========================================================================= #
def verify_output(output_name):
    has_output  = os.path.isfile(output_name + '.npz')
    if has_output:
        question = "**** You already have the file '{}' ****\nDo you REALLY want to REDO the calculations? (y/N)\n".format(output_name)
        default_answer = 'N'
    else:
        question = "**** You do not have the file '{}' yet ****\nDo you want to proceed with calculations NOW? (Y/n)\n".format(output_name)
        default_answer = 'Y'
    answer = input(question) or default_answer
    return answer.upper()

def output_file(output_name, data_dic_to_save):
    current_path = os.getcwd()
    complete_path = current_path + '/' + output_name
    print("\nSaving '%s' ... " % output_name)
    np.savez(complete_path, **data_dic_to_save)
    print("Done!")
