def LevyCov_Min(Spectral_density, Theta, X, alpha, field, nb_it=50):
    """

    Filtering method based on minimization of covariation norm associated to
    a specific spatial representation of an alpha-stable distribution.

    Input
    -----
    Spectral_density: (J, P)
    or "spectrum density"

    Theta: (K, P)
    sample of K-hypersphere

    X :(T, K)
    observations

    alpha: 0<alpha<=2
    characteristic exponent

    Returns:

    Y_hat: (J, T, K)
    estimation of y

    """

    def update_SCM(Spectral_density, Theta, alpha, W, field):
        """

        Args:
            Spectral_density: (J, P)
            or "spectrum density"

            Theta: (K, P)
            sample of K-hypersphere

            alpha: 0<alpha<=2
            characteristic exponent

            W: (J, K, K)
            parameters for optimization

        Returns:
            C (J, K, K, K) Spatial_covariation_matrix

        """

        J = W.shape[0]
        P = Spectral_density.shape[-1]
        K = Theta.shape[0]

        eps = 1e-12

        Spectral_density_X = np.sum(Spectral_density, axis=0)  # Gamma_X (P)

        # (P, K, K) => (Theta x Theta.H)
        if field == 'R':
            Cste = float(np.pi / P)  # integration constant
            Theta_matrices = Theta.T[..., None] * Theta.T[:, None, :]
            W_inner_Theta = np.dot(W, Theta)  # (J, K, P) =>  <w_j,k; theta>

            temp = np.zeros((J, K, P)).astype(np.float128)
            for j in range(J):
                for k in range(K):
                    temp[j, k, :] = Theta[k, :] -\
                                    W_inner_Theta[j, k, :]  # (J, K, P)
        elif field == 'C':
            Cste = float(4 * np.pi / P)  # integration constant
            Theta_matrices = Theta.T.conj()[..., None] * Theta.T[:, None, :]
            # (J, K, P) =>  <w_j,k; theta>
            W_inner_Theta = np.dot(W.conj(), Theta)

            temp = np.zeros((J, K, P)).astype(np.complex64)
            for j in range(J):
                for k in range(K):
                    temp[j, k, :] = Theta[k, :] -\
                                    W_inner_Theta[j, k, :]  # (J, K, P)

        # (J, K, P) => |theta_k - <w_j,k; theta>|^(alpha - 2)
        Den_temp_1 = np.abs(temp) ** (2. - alpha)
        # (J, K, P) => |<w_j,k; theta>|^(alpha - 2)
        Den_temp_2 = np.abs(W_inner_Theta + eps) ** (2. - alpha)

        temp_1 = Spectral_density[:, None, :] / (Den_temp_1 + eps)  # (J, K, P)
        temp_2 = np.array([Spectral_density_X -
                           Spectral_density[j, :] for j in range(J)])  # (J, P)
        temp_3 = temp_2[:, None, :] / (Den_temp_2 + eps)  # (J, K, P)

        C = Cste * np.sum(Theta_matrices[None, None, ...] *
                          (temp_1[..., None, None] +
                           temp_3[..., None, None]), axis=-3)  # (J, K, K, K)

        return C

    def update_W(Spectral_density, Theta, alpha, W, C, Lambda, field):
        """

        Args:
            Spectral_density: (J, P)
            or "spectrum density"

            Theta: (K, P)
            sample of K-hypersphere

            alpha: 0<alpha<=2
            characteristic exponent

            W: (J, K, K)
            parameters for optimization

            C: (J, K, K, K)
            Spatial_covariation_matrix

            Lambda: (K, K)
            Lagrange multiplicative constant

        Returns:
            W: (J, K, K)
            parameters for optimization

        """
        J = C.shape[0]
        P = Spectral_density.shape[-1]

        eps = 1e-12

        if field == 'R':
            Cste = float(np.pi / P)  # integration constant
            C_invert = invert(C, field='real')  # (J, K, K, K)
            W_inner_Theta = np.dot(W, Theta)  # (J, K, P) =>  <w_j,k; theta>

        elif field == 'C':
            Cste = float(4 * np.pi / P)  # integration constant
            C_invert = invert(C, field='complex')  # (J, K, K, K)
            # (J, K, P) =>  <w_j,k; theta>
            W_inner_Theta = np.dot(W.conj(), Theta)

        # (J, K, P) => |theta_k - <w_j,k; theta>|^(alpha - 2)
        temp1 = np.array([Theta - W_inner_Theta[j, ...] for j in range(J)])
        Den_temp = np.abs(temp1) ** (2. - alpha)

        # (J, K, P)
        if field == 'R':
            temp2 = np.einsum('abc,bc->abc', (Den_temp + eps) ** -1, Theta)
        elif field == 'C':
            temp2 = np.einsum('abc,bc->abc', (Den_temp + eps) ** -1,
                              Theta.conj())

        # (J, 1, P) =>  gamma_j(dtheta)
        Num_temp = Spectral_density[:, None, :]
        temp = Num_temp * temp2  # (J, K, P)

        integrand = Theta[None, None, ...] * temp[..., None, :]  # (J, K, K, P)
        temp_integral = Cste * np.sum(integrand, axis=-1)  # (J, K, K)

        temp = np.array([temp_integral[j, ...] -
                         Lambda for j in range(J)])  # (J, K, K)
        temp_final = C_invert * temp[..., None]  # (J, K, K, K)
        new_W = np.sum(temp_final, axis=-2)  # (J, K, K)

        return new_W

    def update_Lambda(C, W, Lambda, field):
        """

        Args:
            C: (J, K, K, K)
            Spatial_covariation_matrix

            W: (J, K, K)
            parameters for optimization

            Lambda: (K, K)
            Lagrange multiplicative constant

        Returns:
            Lambda: (K, K)
            Lagrange multiplicative constant

        """

        K = Lambda.shape[0]

        if field == 'R':
            C_invert = invert(C, field='real')  # (J, K, K, K)
            # (K, K, K) => Sigma_j(P_j,k^-1)^-1
            C_invert = invert(np.sum(C_invert, axis=0), field='real')

        if field == 'C':
            C_invert = invert(C, field='complex')  # (J, K, K, K)
            # (K, K, K) => Sigma_j(P_j,k^-1)^-1
            C_invert = invert(np.sum(C_invert, axis=0), field='complex')

        temp = np.sum(W, axis=0) - np.eye(K)  # (K, K)
        for k in range(K):
            Lambda[k, :] += np.dot(C_invert[k, ...], temp[k, :])  # (K)

        return Lambda

    # Beginning of the algorithm
    J = Spectral_density.shape[0]
    K = X.shape[-1]

    if field == 'R':
        C = np.random.randn(J, K, K, K).astype(np.float128)  # Spatial covariation matrix
        Lambda = np.zeros((K, K)).astype(np.float128)
        W = np.zeros((J, K, K)).astype(np.float128)

    elif field == 'C':
        # Spatial covariation matrix
        C = np.random.randn(J, K, K, K).astype(np.complex64)
        Lambda = np.zeros((K, K)).astype(np.complex64)
        W = np.zeros((J, K, K)).astype(np.complex64)

    for j in range(J):
        for k in range(K):
            W[j, k, :] = sig.unit_impulse(K, k)/J

    # Estimation with covariation

    if alpha <= 1.2:
        alpha = 1.24
    for it in range(nb_it):
        C = update_SCM(Spectral_density, Theta, alpha, W, field)
        Lambda = update_Lambda(C, W, Lambda, field)
        W = update_W(Spectral_density, Theta, alpha, W, C, Lambda, field)

    X = X.T  # (K, T)
    if field == 'R':
        Y = np.dot(W, X)
    if field == 'C':
        Y = np.dot(W.conj(), X)
    Y = np.swapaxes(Y, 1, 2)  # (J, T, K)

    return Y
