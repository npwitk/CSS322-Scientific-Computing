import numpy as np

def qr_iteration(A, tol=1e-6, max_iter=50, verbose=True):
    """
    Perform basic QR iteration to approximate eigenvalues.
    
    Parameters:
        A : np.ndarray
            Initial square matrix.
        tol : float
            Convergence tolerance for off-diagonal elements.
        max_iter : int
            Maximum number of iterations.
        verbose : bool
            If True, prints iteration details.
    """
    A_k = A.copy().astype(float)
    np.set_printoptions(precision=5, suppress=True)  # avoid scientific notation

    for k in range(max_iter):
        Q, R = np.linalg.qr(A_k)
        A_next = R @ Q

        # Round tiny floating-point errors for display
        A_next = np.round(A_next, 5)

        if verbose:
            print(f"\nIteration {k}:")
            print(A_next)

        off_diag_norm = np.sqrt(np.sum(np.tril(A_next, -1)**2 + np.triu(A_next, 1)**2))
        if off_diag_norm < tol:
            print(f"\nConverged after {k} iterations.")
            break

        A_k = A_next

    eigs = np.round(np.diag(A_next), 5)
    print("\nApproximate eigenvalues:", eigs)
    return A_next, eigs


if __name__ == "__main__":
    A = np.array([[4, -1],
                  [-1, 6]], dtype=float)
    
    print("Initial Matrix A:")
    print(A)
    print("\nStarting QR Iteration...\n")

    final_A, eigenvalues = qr_iteration(A)
