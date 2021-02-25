
import numpy as np
import os, shutil, subprocess
import hamiltonians as ham

# ========================================================================= #
##                          READ/GENERATE INPUT FILE
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

def decide_ham():
    question = "Which Hamiltonian do you want to use: default == 4 "
    options  = "\n (1) H2x2 \n (2) H4x4 \n (3) H4x4_equal \n (4) H4x4_Kormanyos_Fabian \n"
    default  = '4'
    answer   = input(question + options) or default
    if int(answer) == 4:
        main_input_file_template = 'infile_Kormanyos_Fabian.txt'
    else:
        main_input_file_template = 'infile.txt'
    return main_input_file_template

def read_params(specific_file):
    """
    This function open a 'specific_file', reads and parses the
    information contained in the input file.

    The formatation of the file must follow the structure:

    Ham        = Hclass  ## it doesn't need a type.
    param_name = value   ## type_of_param : comment

    The output of this function will be a dictionary with all the params
    already converted to its correct type.

    """
    current_path = os.getcwd()
    file  = open(specific_file,'r');
    lines = file.readlines()
    file.close()
    params = {}

    for line in lines:
        list_line = line.split()
        if len(list_line) == 0 or list_line[0]=='##': continue
        elif list_line[0] == 'Ham': params[list_line[0]] = eval('ham.'+list_line[2])
        else: params[list_line[0]] = eval(list_line[4])(list_line[2])

    return params

def read_file(specific):
    """
    # TODO: DOCSTRING
    """
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
    # if not has_input and name_file == 'infile.txt':
    #     template_name = decide_ham()
    #     open_template(template_name)
    # elif not has_input:
    #     open_template(name_file)
    if not has_input:
        open_template(name_file)
    else :
        open_infile(name_file)

def from_dic_to_var(Ham, gamma, Egap, r0, mc, mv, alpha_option, epsilon, exchange, d_chosen, L_k, n_mesh, n_sub, submesh_radius, n_rec_states):

    ## MODEL HAMILTONIAN
    Ham           = eval('ham.' + Ham)  # chosen Hamiltonian has to be a class in 'hamiltonians.py'
    gamma         = float(gamma)        # meV*nm
    Egap          = float(Egap)         # meV
    r0            = float(r0)           # nm
    mc            = float(mc)           # conduction band effective mass
    mv            = float(mv)           # valence band effective mass
    alpha_option  = str(alpha_option)   # options: (zero, masses, corrected)
    epsilon       = float(epsilon)      # effective dielectric constant

    ## BSE EXCHANGE:
    exchange      = int(exchange)                   # Boll: 0 = False, 1 = True
    d_chosen      = float(d_chosen)                 # Chose a value in the interval [60, 65] nm

    ## K-SPACE
    L_k             = float(L_k)                # the discretized k-space limits: k_x(y) ∈ [-L,L]
    n_mesh          = int(n_mesh)               # quantity of points in each direction
    n_sub           = int(n_sub)                # quantity of points in each of the submesh direction
    submesh_radius  = int(submesh_radius)       # in number of sites; radius of the region where the submesh will be considered

    ## SAVING OPTIONS:
    n_rec_states    = int(n_rec_states)         # number of states to be recorded

    return Ham, gamma, Egap, r0, mc, mv, alpha_option, epsilon, exchange, d_chosen, L_k, n_mesh, n_sub, submesh_radius, n_rec_states

def from_dic_to_var_kormanyos_fabian(Ham, E_c, E_v, alpha_up, alpha_dn, beta_up, beta_dn, gamma, delta_c, delta_v, kappa_up, kappa_dn, valey, epsilon, r0, exchange, d_chosen, L_k, n_mesh, n_sub, submesh_radius, n_rec_states):
    ## MODEL HAMILTONIAN
    Ham         = eval('ham.' + Ham)     # chosen Hamiltonian has to be a class in 'hamiltonians.py'
    E_c         = 1e3 * float(E_c)       # H_0   [eV]   --> [meV]
    E_v         = 1e3 * float(E_v)       # H_0   [eV]   --> [meV]
    alpha_up    = 10 * float(alpha_up)   # H_2kp [eV Å²]--> [meV nm²]
    alpha_dn    = 10 * float(alpha_dn)   # H_2kp [eV Å²]--> [meV nm²]
    beta_up     = 10 * float(beta_up)    # H_2kp [eV Å²]--> [meV nm²]
    beta_dn     = 10 * float(beta_dn)    # H_2kp [eV Å²]--> [meV nm²]
    gamma       = 1e2 * float(gamma)     # H_1kp [eV Å] --> [meV nm]
    delta_c     = 1e3 * float(delta_c)   # H_SO  [eV]   --> [meV]
    delta_v     = 1e3 * float(delta_v)   # H_SO  [eV]   --> [meV]
    kappa_up    = 10 * float(kappa_up)   # H_2kp [eV Å²]--> [meV nm²]
    kappa_dn    = 10 * float(kappa_dn)   # H_2kp [eV Å²]--> [meV nm²]
    valey       = int(valey)             # 1 == K-point; 1 == K'-point
    epsilon     = float(epsilon)         # effective dielectric constant
    r0          = 1e-1 * float(r0)          # [Å] --> [nm]
    ## BSE EXCHANGE:
    exchange      = int(exchange)        # Boll: 0 = False, 1 = True
    d_chosen      = float(d_chosen)      # Chose a value in the interval [60, 65] nm
    ## K-SPACE
    L_k             = float(L_k)                # the discretized k-space limits: k_x(y) ∈ [-L,L]
    n_mesh          = int(n_mesh)               # quantity of points in each direction
    n_sub           = int(n_sub)                # quantity of points in each of the submesh direction
    submesh_radius  = int(submesh_radius)       # in number of sites; radius of the region where the submesh will be considered
    ## SAVING OPTIONS:
    n_rec_states    = int(n_rec_states)         # number of states to be recorded

    return Ham, E_c, E_v, alpha_up, alpha_dn, beta_up, beta_dn, gamma, delta_c, delta_v, kappa_up, kappa_dn, valey, epsilon, r0, exchange, d_chosen, L_k, n_mesh, n_sub, submesh_radius, n_rec_states

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

def pop_out_model(params_dictionary):
    ## HAMILTONIAN:
    H = params_dictionary.pop('Ham')
    ## POTENTIAL:
    r0      = params_dictionary.pop('r0')
    epsilon = params_dictionary.pop('epsilon')
    ## EXCHANGE:
    exchange = bool(params_dictionary.pop('exchange'))
    d_value  = params_dictionary.pop('d_chosen')
    ## GRID:
    Lk   = params_dictionary.pop('L_k')
    Nk   = params_dictionary.pop('n_mesh')
    Nsub = params_dictionary.pop('n_sub')
    Rsub = params_dictionary.pop('submesh_radius')
    ## SAVING OPTIONS
    Nsaved = params_dictionary.pop('n_rec_states')
    return H, r0, epsilon, exchange, d_value, Lk, Nk, Nsub, Rsub, Nsaved




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
