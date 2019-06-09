import numpy as np
import cv2

# Image Smoothing via L0 Gradient Minimization
def L0Smooth(img, lam = 2e-2, kappa = 2):
    betaMax = 1e5
    img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    mat = np.array(img)
    N, M, D = img.shape
    sizeI2D = [N, M]
    fx = np.array([[1, -1]])
    fy = np.array([[1], [-1]])
    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)
    Normin1 = np.fft.fft2(mat, axes=(0,1))
    Denormin2 = abs(otfFx) ** 2 + abs(otfFy) ** 2
    if(D > 1):
        Denormin2 =  np.tile(Denormin2.reshape((N, M, 1)), (1, 1, D))
    beta = 2 * lam
    while(beta < betaMax):
        Denormin = 1 + beta * Denormin2
        # h-v sub-problem
        h1 = np.diff(mat, 1, 1)
        h2 = np.reshape(mat[:, 0], (N, 1, D)) - np.reshape(mat[:, -1], (N, 1, D))
        h = np.hstack((h1, h2))

        v1 = np.diff(mat, 1, 0)
        v2 = np.reshape(mat[0, :], (1, M, D)) - np.reshape(mat[-1, :], (1, M, D))
        v = np.vstack((v1, v2))
        if (D == 1):
            t = (h**2 + v**2) < lam / beta
        else:
            t = np.sum((h ** 2 + v ** 2), 2) < lam / beta
            t = np.tile(t.reshape(N, M, 1), (1, 1, D))

        h[t] = 0
        v[t] = 0

        # S sub-problem
        h1 = np.reshape(h[:, -1], (N, 1, D)) - np.reshape(h[:, 0], (N, 1, D))
        h2 = -np.diff(h, 1, 1)
        Normin2 = np.hstack((h1, h2))
        v1 = np.reshape(v[-1, :], (1, M, D)) - np.reshape(v[0, :], (1, M, D))
        v2 = -np.diff(v, 1, 0)
        Normin2 += np.vstack((v1, v2))
        FS = (Normin1 + beta*np.fft.fft2(Normin2, axes=(0,1))) / Denormin
        mat = np.real(np.fft.ifft2(FS, axes=(0,1)))
        beta *= kappa
    return mat

def psf2otf(psf, shape):
    if(np.all(psf == 0)):
        return np.zeros_like(psf)
    psf = padArray(psf, shape)
    # Circularly shift otf so that the "center" of the PSF is at the
    # [0][0] element of the array
    for axis, axis_size in enumerate(psf.shape):
        psf = np.roll(psf, -int(axis_size / 2), axis = axis)

    # fft
    otf = np.fft.fftn(psf)
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)
    return otf

# padarray function in matlab with padval = 'post'
def padArray(a, shape):
    padShape = np.asarray(shape) - np.asarray(a.shape)
    if(np.any(padShape) < 0):
        print("negative shape")
    padA = np.zeros(shape, dtype = a.dtype)
    idx, idy = np.indices(np.asarray(a.shape, dtype = int))
    padA[idx, idy] = a
    return padA