"""
latex_formatting_utils.py

Utilities for formatting quantum chemistry data (matrices, eigenvalues, eigenvectors)
into LaTeX format for publication-ready documents.
"""

import numpy as np
from typing import Tuple, List, Optional, Union


def format_complex_coefficient(coeff: complex, threshold: float = 1e-10) -> str:
    """
    Format a complex coefficient for LaTeX display.
    
    Parameters:
    -----------
    coeff : complex
        The coefficient to format
    threshold : float
        Threshold below which values are considered zero
    
    Returns:
    --------
    str : Formatted coefficient string
    """
    real, imag = np.real(coeff), np.imag(coeff)
    
    # Special cases for ±1, ±i
    if np.abs(coeff - 1) < threshold:
        return ""
    elif np.abs(coeff + 1) < threshold:
        return "-"
    elif np.abs(coeff - 1j) < threshold:
        return "i"
    elif np.abs(coeff + 1j) < threshold:
        return "-i"
    
    # Pure real or imaginary
    if np.abs(imag) < threshold:
        return f"{real:.4f}"
    elif np.abs(real) < threshold:
        return f"{imag:.4f}i"
    else:
        sign = "+" if imag >= 0 else "-"
        return f"({real:.4f} {sign} {abs(imag):.4f}i)"


def binary_to_ket(index: int, n_qubits: int = 4) -> str:
    """
    Convert basis state index to ket notation using \\ket{} command.
    
    Assumes \\ket{} is defined in LaTeX preamble (e.g., via braket package
    or custom definition like \\newcommand{\\ket}[1]{|#1\\rangle}).
    
    Parameters:
    -----------
    index : int
        Basis state index (0 to 2^n_qubits - 1)
    n_qubits : int
        Number of qubits
    
    Returns:
    --------
    str : LaTeX ket string like \\ket{0000}
    """
    binary = format(index, f'0{n_qubits}b')
    return f"\\ket{{{binary}}}"


def binary_to_bra(index: int, n_qubits: int = 4) -> str:
    """
    Convert basis state index to bra notation using \\bra{} command.
    
    Assumes \\bra{} is defined in LaTeX preamble (e.g., via braket package
    or custom definition like \\newcommand{\\bra}[1]{\\langle#1|}).
    
    Parameters:
    -----------
    index : int
        Basis state index (0 to 2^n_qubits - 1)
    n_qubits : int
        Number of qubits
    
    Returns:
    --------
    str : LaTeX bra string like \\bra{0000}
    """
    binary = format(index, f'0{n_qubits}b')
    return f"\\bra{{{binary}}}"


def eigenvector_to_dirac(eigenvector: np.ndarray, 
                         threshold: float = 1e-3, 
                         max_terms: int = 6,
                         use_spaces: bool = True) -> str:
    """
    Convert eigenvector to Dirac notation using \\ket{} commands.
    
    Parameters:
    -----------
    eigenvector : numpy.ndarray
        The eigenvector to convert
    threshold : float
        Minimum coefficient magnitude to include
    max_terms : int
        Maximum number of terms to show
    use_spaces : bool
        If True, add spaces between terms for readability
    
    Returns:
    --------
    str : LaTeX formatted sum of kets
    """
    n_qubits = int(np.log2(len(eigenvector)))
    
    # Get significant components
    components = [(i, coeff) for i, coeff in enumerate(eigenvector) 
                  if np.abs(coeff) > threshold]
    components.sort(key=lambda x: -np.abs(x[1]))  # Sort by magnitude
    
    terms = []
    for i, (idx, coeff) in enumerate(components[:max_terms]):
        ket = binary_to_ket(idx, n_qubits)
        coeff_str = format_complex_coefficient(coeff)
        
        # Format the term with proper sign handling
        if i == 0:
            term = f"{coeff_str}{ket}"
        else:
            if coeff_str.startswith('-'):
                term = f"- {coeff_str[1:]}{ket}" if use_spaces else f"-{coeff_str[1:]}{ket}"
            else:
                term = f"+ {coeff_str}{ket}" if use_spaces else f"+{coeff_str}{ket}"
        
        terms.append(term)
    
    if len(components) > max_terms:
        terms.append("+ \\cdots" if use_spaces else "+\\cdots")
    
    separator = " " if use_spaces else ""
    return separator.join(terms)


def eigenvector_to_bra(eigenvector: np.ndarray, 
                        threshold: float = 1e-3, 
                        max_terms: int = 6,
                        use_spaces: bool = True) -> str:
    """
    Convert eigenvector to bra notation (conjugate).
    
    Parameters:
    -----------
    eigenvector : numpy.ndarray
        The eigenvector to convert (will be conjugated)
    threshold : float
        Minimum coefficient magnitude to include
    max_terms : int
        Maximum number of terms to show
    use_spaces : bool
        If True, add spaces between terms
    
    Returns:
    --------
    str : LaTeX formatted sum of bras
    """
    conjugated = np.conj(eigenvector)
    result = eigenvector_to_dirac(conjugated, threshold, max_terms, use_spaces)
    return result.replace("\\ket", "\\bra")


def format_matrix_latex(matrix: np.ndarray, 
                       precision: int = 6,
                       environment: str = "pmatrix") -> str:
    """
    Format a numpy array as LaTeX matrix with controlled precision.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        The matrix to format
    precision : int
        Number of decimal places to display
    environment : str
        LaTeX matrix environment ('pmatrix', 'bmatrix', 'matrix', etc.)
    
    Returns:
    --------
    str : LaTeX formatted matrix string
    """
    n = matrix.shape[0]
    
    # Start the LaTeX matrix
    latex_str = f"\\begin{{{environment}}}"
    
    for i in range(n):
        row_str = []
        for j in range(n):
            val = np.real(matrix[i, j])
            if np.abs(val) < 1e-10:
                row_str.append("0")
            elif np.abs(val - np.round(val)) < 1e-10:
                # Integer value
                row_str.append(f"{int(np.round(val))}")
            else:
                # Format with specified precision
                row_str.append(f"{val:.{precision}f}")
        
        latex_str += " & ".join(row_str)
        if i < n - 1:
            latex_str += r" \\ "
    
    latex_str += f"\\end{{{environment}}}"
    return latex_str


def create_eigenvalue_table(eigenvalues: np.ndarray, 
                           eigenvectors: np.ndarray,
                           n_states: Optional[int] = None,
                           threshold: float = 0.01,
                           max_terms: int = 3,
                           caption: str = "Eigenvalues and eigenvectors") -> str:
    """
    Create a LaTeX table of eigenvalues and eigenvectors.
    
    Parameters:
    -----------
    eigenvalues : numpy.ndarray
        Array of eigenvalues
    eigenvectors : numpy.ndarray
        Matrix of eigenvectors (columns)
    n_states : int, optional
        Number of states to include (default: all)
    threshold : float
        Minimum coefficient magnitude for eigenvector components
    max_terms : int
        Maximum number of terms per eigenvector
    caption : str
        Table caption
    
    Returns:
    --------
    str : Complete LaTeX table
    """
    if n_states is None:
        n_states = len(eigenvalues)
    else:
        n_states = min(n_states, len(eigenvalues))
    
    latex_table = f"{caption}:\n"
    latex_table += r"""\begin{center}
        \begin{tabular}{ |c|c|c| } 
        \hline
        State & Energy (Hartree) & Eigenvector \\ 
        \hline"""
    
    for i in range(n_states):
        E = eigenvalues[i]
        psi_latex = eigenvector_to_dirac(eigenvectors[:, i], 
                                         threshold=threshold, 
                                         max_terms=max_terms,
                                         use_spaces=False)
        latex_table += f"\n $E_{{{i}}}$ & ${E:.10f}$ & ${psi_latex}$ \\\\"
    
    latex_table += r""" 
        \hline
        \end{tabular}
        \end{center}"""
            
    return latex_table


def format_pauli_hamiltonian(pauli_strings: List[str], 
                            coefficients: List[float],
                            terms_per_line: int = 3) -> str:
    """
    Format a Pauli Hamiltonian in LaTeX.
    
    Parameters:
    -----------
    pauli_strings : list of str
        List of Pauli strings (e.g., ["IIII", "ZIII", ...])
    coefficients : list of float
        Corresponding coefficients
    terms_per_line : int
        Number of terms per line in align environment
    
    Returns:
    --------
    str : LaTeX formatted Hamiltonian
    """
    latex_str = "\\begin{align}\nH = &"
    
    terms = []
    for pauli, coeff in zip(pauli_strings, coefficients):
        if np.abs(coeff) < 1e-10:
            continue
        
        # Format coefficient
        if coeff > 0:
            sign = "+"
        else:
            sign = "-"
        
        # Format Pauli string
        pauli_latex = "".join([f"\\sigma_{i}^{p.lower()}" if p != 'I' else "\\mathbb{I}_{" + str(i) + "}"
                               for i, p in enumerate(pauli)])
        pauli_latex = pauli_latex.replace("\\sigma", "\\sigma")  # Cleanup
        
        # Simple formatting for now
        terms.append(f"{sign} {abs(coeff):.6f} {pauli}")
    
    # Group terms
    for i, term in enumerate(terms):
        if i == 0 and term.startswith("+"):
            term = term[1:].strip()
        
        if i > 0 and i % terms_per_line == 0:
            latex_str += " \\\\\n& "
        
        latex_str += f" {term}"
    
    latex_str += "\n\\end{align}"
    return latex_str


def save_latex_document(content: str, 
                        filename: str = "output.tex",
                        preamble: Optional[str] = None) -> None:
    """
    Save content as a complete LaTeX document.
    
    Parameters:
    -----------
    content : str
        Main content to include in document
    filename : str
        Output filename
    preamble : str, optional
        Custom preamble (default includes basic packages)
    """
    if preamble is None:
        preamble = r"""\documentclass[11pt]{article}
        \usepackage{amsmath}
        \usepackage{braket}
        \usepackage{booktabs}
        \usepackage{geometry}
        \geometry{margin=1in}

        \title{Quantum Chemistry Calculations}
        \date{\today}
        """
    
    full_document = preamble + "\n\\begin{document}\n\\maketitle\n\n"
    full_document += content
    full_document += "\n\n\\end{document}"
    
    with open(filename, 'w') as f:
        f.write(full_document)
    
    print(f"LaTeX document saved to {filename}")


# Example usage function
def example_usage():
    """Demonstrate usage of the formatting utilities."""
    # Create a sample eigenvector
    eigenvector = np.array([0.9936, 0, -0.1128, 0, 0.0056, 0, 0.0001, 0,
                            0, 0, 0, 0, 0, 0, 0, 0])
    
    # Format as Dirac notation
    dirac = eigenvector_to_dirac(eigenvector, threshold=0.001)
    print(f"Eigenvector in Dirac notation: {dirac}")
    
    # Create sample matrix
    matrix = np.diag([1, 2, 3, 4])
    matrix_latex = format_matrix_latex(matrix, precision=2)
    print(f"\nMatrix in LaTeX:\n{matrix_latex}")
    
    # Create eigenvalue table
    eigenvalues = np.array([-1.851, -1.246, -0.475, 0.206])
    eigenvectors = np.eye(4)
    table = create_eigenvalue_table(eigenvalues, eigenvectors, n_states=4)
    print(f"\nEigenvalue table:\n{table}")