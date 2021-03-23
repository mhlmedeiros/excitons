# Bethe-Salpeter Equation code

## Tests

1. Using effective masses: ✅
    * $E_{gap} = 0$
    * $γ = 0$
    * $ε = 1$
    * $m_c = 0.2834$
    * $m_v = -0.3636$
    * $α_c = \frac{1}{m_c}$
    * $α_v = \frac{1}{m_v}$

2. Using couplings but no masses: ✅
    * $E_{gap} = 2.4$ eV
    * $γ = 2.6$ eV ⋅ Å
    * $ε = 1$
    * $α_c = 0$
    * $α_v = 0$

3. Using couplings but no masses (different $ε$): ✅
    * $E_{gap} = 2.4$ eV
    * $γ = 2.6$ eV ⋅ Å
    * $ε = 4.5$
    * $α_c = 0$
    * $α_v = 0$

4. Using couplings corrected masses: ✅
    * $E_{gap} = 2.4$ eV
    * $γ = 2.6$ eV ⋅ Å
    * $ε = 1$ (💡)
    * $α_c = \frac{1}{m_c} + \frac{2m_0}{ħ^2}(\frac{γ}{E_{gap}})$
    * $α_v = \frac{1}{m_c} - \frac{2m_0}{ħ^2}(\frac{γ}{E_{gap}})$

5. Using couplings corrected masses (different $ε$): ✅
    * $E_{gap} = 2.4$ eV
    * $γ = 2.6$ eV ⋅ Å
    * $ε = 4.5$ (💡)
    * $α_c = \frac{1}{m_c} + \frac{2m_0}{ħ^2}(\frac{γ}{E_{gap}})$
    * $α_v = \frac{1}{m_c} - \frac{2m_0}{ħ^2}(\frac{γ}{E_{gap}})$

6. Paulo's test:
    * $E_{gap} = 1311.79$ meV
    * $γ = 3.91504469 \times 10^2$ meV nm
    * $ε = 1$ ◀
    * $α_c = 0$
    * $α_v = 0$
    * mesh = $101\times101$
    * submesh = $101\times101$

---

## Improvements in the code


What I have to do to improve my code:

1. Implement a way to read a file (or files) of parameters
    * The file(s) will contain:
        - Model parameters
        - Reciprocal space information
    * Maybe having two separated files: "model" and "space"
2. Implement an automatic to named the output files **OK**
3. Change the strategy of calculate all eigenvalues to *Lanczos algorithm*.
3. Perform averages over sub-grids around the mesh points in k-space to improve convergence
    * Doing so we'll could abandon the extrapolation strategy **OK**

## Reading the files

Firstly let's see how one the typical physical model is presented. Currently the tests
have been made using the following model

$$
H = \begin{bmatrix}
    \Delta + \alpha_c\frac{\hbar^2}{2m_0} & \gamma (k_x + ik_y) \\
    \gamma (k_x - ik_y) & \alpha_v\frac{\hbar^2}{2m_0}
\end{bmatrix}
$$

with one conduction band and one valence band. But this is not the ultimately
aimed situation. In the future we want to use models with more bands, so the
information about how many conduction and valence bands we have in the model
have to be an input.

The shape of the Hamiltonian, the couplings and the momentum dependence, is
obviously an important issue. There are at least two option to inform this:

1. The simplest way is to have the models predefined in the script
    - leaving only the values for the parameters for the file
2. The more sophisticated way will be to parse a symbolic definition
    - here the whole matrix have to passed by the file and the script have to parse it using `simpy`

We can combine both ways listed above in the following manner:

* Implementing a module containing all the models
* Passing the desired function using the input file: only have to parse the name of the function.

The `model_input_file` will be structured as we see bellow:

```python

# Input model file

# Model function:
hamiltonian_2x2

# Parameters:
2.6 # Egap [eV]
2.4 # Gamma [eV*AA]
0   # alpha_c
0   # alpha_v
```

## Sub-mesh averages

**Tests**

1. Sub-mesh only around the $|\vec{k} - \vec{k}'| = 0$:
    * $N_{sub}$ = 201 x 201
    * Time spent on BSE-matrix construction = 1204.3 s = 20.0 min
    * Total time spent = 1905.6 s = 31.8 min
    * $E_B^{(1s)}$ = 447.57 meV
    * $E_B^{(2s)}$ = 201.72 meV

2. Sub-mesh for all values of $|\vec{k} - \vec{k}'|$  
    * $N_{sub}$ = 101 x 101
    * Time spent on BSE-matrix construction = 5004.3 s = 83.4 min
    * Total time spent = 5665.8 s = 94.4 min
    * $E_B^{(1s)}$ = 454.68 meV
    * $E_B^{(2s)}$ = 207.58 meV



## Results

### BUG (FIXED)!!
For Hamiltonian models with a non-zero coupling between the bands something
weird have been happening: The ground state of the system is a $2p_x$-state instead
of a $1s$-state. Since the major difference between these systems are the presence
of a non-trivial $\Delta_{k_i}^{k_j}$, my first guess is that some semantic error
may be occurring.

$$[E_c(k) - E_v(k) - \Omega_N]\mathcal{A}^{(N)}_{cvk} + \sum_{c'v'k'} \mathbb{D}_{c'v'k'}^{cvk} \mathcal{A}^{(N)}_{c'v'k'}=0$$

$$\mathbb{D}_{c'v'k'}^{cvk} = -\Delta_{c'v'k'}^{cvk} V(k-k')$$

$$\Delta_{c'v'k'}^{cvk} = [\sum^M_l \bar{\beta}_{c,l}(k)\beta_{c',l}(k')][\sum^M_l \bar{\beta}_{v',m}(k')\beta_{v,m}(k)]$$

$$\Delta_{c'v'k'}^{cvk} = \langle\phi_{c}(k)|\phi_{c'}(k')\rangle
~\langle\phi_{v'}(k')|\phi_{v}(k)\rangle$$

**SOLUTION:**
For some fucking reason I was diagonalizing only the real part
of the BSE-matrix (😠!).

**Aftermath:**

* The first test results (with effective masses and without couplings) are correct, because in that case we actually have a real matrix.

* The other tests need to be re-run: due to existence of complex matrix-elements in BSE-matrix (😞).    

---


## Speeding up the code

### The Building problem (Numba $\rightarrow$ Fortran)

In my personal PC (where I can install any package I want) I've been using the Python-package named **Numba**. In summary, it takes some selected functions (selected by the developer using decorators) and perform what is called just in time (JIT) compilation. It really helpful but... it is not installed in the university clusters.

One possible solution is to rewrite the computational intensive functions into Fortran procedures (functions, subroutines, modules) and translate it back to python using **f2py**. This was the strategy adopted to run the excitons code using the Wannier Hamiltonian.

To "compilate" the Fortran module and turn it a importable python module I used the following command-line:

`f2py -c -m potential rytova_keldysh.f95`

To call the content in the python code use

```python
import potential
V_single = potential.potential_average(kx, ky, dk, Nsub, eps, r0)
V_total  = potential.build_potential_matrix(kx_flat, ky_flat,
                                            N_total, N_sub,
                                            eps, r_0)
```
**Actually it is safer to call:**

```python
import potential
print(dir(potential))
print(potential.build_potential_matrix.__doc__)
print(potential.potential_average.__doc__)
```

**to see the exactly the arguments required in each function.**

Some times I compile using the following command:

`f2py -c -m potential rytova_keldysh.f95 only: build_potential_matrix`

to make only the `build_potential_matrix` visible to the python code.

Notice, however, this Fortran module (`rytova_keldysh.f95`) solve only the problem with the potential. The mixing term ($\Delta$'s) and the exchange term have not been implemented yet. 😅

## The diagonalization part of the fight

Once the building part of the code ends, the next battle is in the field of eigenvalues and eigenvalues. To solve for the eigenstuff I've been using the `numpy.eigh()` function. It is very good tool for most of the situations, but in that occasions where one only needs the first (smallest) eigenvalues it turns out to be a overkill option.

The smartest option is to use something like the Lanczos algorithm or (more generally) the Arnoldi algorithm. There is a package written in Fortran that does exactly that (ARPACK) and it is present in scipy (Got sai danke). However, before using that it is necessary to transform the dense matrix into a sparse one. So the workflow goes like that:

```python
from scipy import sparse
...
H               = foo(**args)            # function that generate a dense_matrix
sH              = sparse.csr_matrix(H)   # convert into a sparse matrix
Values, Vectors = sparse.linalg.eigs(A, k=6, which='LM')
```

`which` **k eigenvectors** and eigenvalues to find:

* ‘LM’ : largest magnitude
* ‘SM’ : smallest magnitude
* ‘LR’ : largest real part
* ‘SR’ : smallest real part
* ‘LI’ : largest imaginary part
* ‘SI’ : smallest imaginary part
