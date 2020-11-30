import sys
import models

def read_files():
    try:
        model_input_file = sys.argv[1];
        space_input_file = sys.argv[2]
    except:
        print("Usage: ", sys.argv[0]," model_file k_space_file"); sys.exit(1)

    with open(model_input_file) as model:
        lines_without_comments = (line.split('#', 1)[0] for line in model)
        lines_without_spaces = (line.rstrip() for line in lines_without_comments)
        lines_model_nonempty = list(line for line in lines_without_spaces if line)
        # count = 0
        # for l in lines_model_nonempty: print("line %d : %s " %(count, l)); count +=1

    with open(space_input_file) as space:
        lines_without_comments = (line.split('#', 1)[0] for line in space)
        lines_without_spaces = (line.rstrip() for line in lines_without_comments)
        lines_space_nonempty = list(line for line in lines_without_spaces if line)
        # count = 0
        # for l in lines_space_nonempty: print("line %d : %s " %(count, l)); count +=1

    return lines_model_nonempty, lines_space_nonempty


def organize_functions_and_params(lines_model, lines_space):
    Hamiltonian = eval("models." + lines_model_nonempty[0])
    Potential = eval("models." + lines_space_nonempty[0])
    
    return Hamiltonian, Potential, params_Hamiltonian, params_Potential, params_space

def main():
    read_files()

if __name__ == '__main__':
    main()
