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
    * Doing so we'll could abandon the extrapolation strategy

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
