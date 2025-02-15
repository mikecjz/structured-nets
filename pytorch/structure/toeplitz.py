'''Functions to multiply by a Toeplitz-like matrix.
'''
import numpy as np
import torch

from .complex_utils import complex_mult, conjugate
from .krylov import Krylov


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##### Fast multiplication for the Toeplitz-like case

def toeplitz_krylov_transpose_multiply(v, u, f=0.0):
    """Multiply Krylov(Z_f, v_i)^T @ u.
    Parameters:
        v: (rank, n)
        u: (batch_size, n)
        f: real number
    Returns:
        product: (batch, rank, n)
    """
    _, n = u.shape
    _, n_ = v.shape
    assert n == n_, 'u and v must have the same last dimension'
    if f != 0.0:  # cycle version
        # Computing the roots of f
        mod = abs(f) ** (torch.arange(n, dtype=u.dtype, device=u.device) / n)
        if f > 0:
            arg = torch.stack((torch.ones(n, dtype=u.dtype, device=u.device),
                               torch.zeros(n, dtype=u.dtype, device=u.device)), dim=-1)
        else:  # Find primitive roots of -1
            angles = torch.arange(n, dtype=u.dtype, device=u.device) / n * np.pi
            arg = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        eta = mod[:, np.newaxis] * arg
        eta_inverse = (1.0 / mod)[:, np.newaxis] * conjugate(arg)

        u_temp = eta_inverse * u[..., np.newaxis]
        u_temp = torch.complex(u_temp[..., 0], u_temp[..., 1])
        u_f = torch.fft.ifft(u_temp)

        v_temp = eta * v[..., np.newaxis]
        v_temp = torch.complex(v_temp[..., 0], v_temp[..., 1])
        v_f = torch.fft.fft(v_temp)

        uv_f = u_f[:, np.newaxis]* v_f[np.newaxis]
        uv = torch.fft.ifft(uv_f)
        # We only need the real part of complex_mult(eta, uv)
        eta_temp = torch.complex(eta[..., 0], eta[..., 1])
        return torch.real(eta_temp * uv)
    else:
        u_f_temp = torch.fft.rfft(torch.cat((u.flip(1), torch.zeros_like(u)), dim=-1))
        u_f = torch.stack((torch.real(u_f_temp), torch.imag(u_f_temp)), dim=-1)

        v_f_temp = torch.fft.rfft(torch.cat((v, torch.zeros_like(v)), dim=-1))
        v_f = torch.stack((torch.real(v_f_temp), torch.imag(v_f_temp)), dim=-1)

        uv_f_temp = complex_mult(u_f[:, np.newaxis], v_f[np.newaxis])
        uv_f = torch.complex(uv_f_temp[..., 0], uv_f_temp[..., 1])

        return torch.fft.irfft(uv_f, n =2 * n)[..., :n].flip(2)
 

def toeplitz_krylov_multiply_by_autodiff(v, w, f=0.0):
    """Multiply \sum_i Krylov(Z_f, v_i) @ w_i, using Pytorch's autodiff.
    This function is just to check the result of toeplitz_krylov_multiply.
    Parameters:
        v: (rank, n)
        w: (batch_size, rank, n)
        f: real number
    Returns:
        product: (batch, n)
    """
    batch_size, rank, n = w.shape
    rank_, n_ = v.shape
    assert n == n_, 'w and v must have the same last dimension'
    assert rank == rank_, 'w and v must have the same rank'

    u = torch.zeros((batch_size, n), dtype=v.dtype, device=v.device, requires_grad=True)
    prod = toeplitz_krylov_transpose_multiply(v, u, f)
    result, = torch.autograd.grad(prod, u, grad_outputs=w, create_graph=True)
    return result


def toeplitz_krylov_multiply(v, w, f=0.0):
    """Multiply \sum_i Krylov(Z_f, v_i) @ w_i.
    Parameters:
        v: (rank, n)
        w: (batch_size, rank, n)
        f: real number
    Returns:
        product: (batch, n)
    """
    _, rank, n = w.shape
    rank_, n_ = v.shape
    assert n == n_, 'w and v must have the same last dimension'
    assert rank == rank_, 'w and v must have the same rank'
    if f != 0.0:  # cycle version
        # Computing the roots of f
        mod = abs(f) ** (torch.arange(n, dtype=w.dtype, device=w.device) / n)
        if f > 0:
            arg = torch.stack((torch.ones(n, dtype=w.dtype, device=w.device),
                               torch.zeros(n, dtype=w.dtype, device=w.device)), dim=-1)
        else:  # Find primitive roots of -1
            angles = torch.arange(n, dtype=w.dtype, device=w.device) / n * np.pi
            arg = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        eta = mod[:, np.newaxis] * arg
        eta_inverse = (1.0 / mod)[:, np.newaxis] * conjugate(arg)
        temp_w = eta * w[..., np.newaxis]
        temp_w = torch.complex(temp_w[..., 0], temp_w[..., 1])
        w_f = torch.fft.fft(temp_w)
        temp_v = eta * v[..., np.newaxis]
        temp_v = torch.complex(temp_v[..., 0], temp_v[..., 1])
        v_f = torch.fft.fft(temp_v)
        wv_sum_f = (w_f * v_f).sum(dim=1)
        wv_sum = torch.fft.ifft(wv_sum_f)
        # We only need the real part of complex_mult(eta_inverse, wv_sum)
        eta_inverse_temp = torch.complex(eta_inverse[..., 0], eta_inverse[..., 1])
        return torch.real(eta_inverse_temp * wv_sum)
    else:
        w_f_temp = torch.fft.rfft(torch.cat((w, torch.zeros_like(w)), dim=-1))
        v_f_temp = torch.fft.rfft(torch.cat((v, torch.zeros_like(v)), dim=-1))

        w_f = torch.stack((torch.real(w_f_temp), torch.imag(w_f_temp)), dim=-1)
        v_f = torch.stack((torch.real(v_f_temp), torch.imag(v_f_temp)), dim=-1)

        wv_sum_f_temp = complex_mult(w_f, v_f).sum(dim=1)
        wv_sum_f = torch.complex(wv_sum_f_temp[..., 0], wv_sum_f_temp[..., 1])

        return torch.fft.irfft(wv_sum_f, n = 2 * n)[..., :n]
    
def toeplitz_transpose_multiply_fft(H, u, dim = 1, is_complex=False):
    """Multiply U(v) @ w (where U is the upper-triangular Toeplitz matrix) using FFT.
    Parameters:
        H: 1D: (rank, n) or 2D: (rank, n, n)
        u: 1D: (batch_size, n) or 2D: (batch_size, rank, n, n)
        dim: whether to perform 1D or 2D multiplication
    Returns:
        product: (batch, rank, n)
    """
    n = H.shape[1]
    
    # 1D case
    if dim == 1:
        # zero-pad H and w to 2n
        H = torch.cat((H, torch.zeros_like(H)), dim=-1) # (rank, 2n)
        u = torch.cat((u, torch.zeros_like(u)), dim=-1) # (batch_size, 2n)
        
        H_f = torch.fft.fft(H) # (rank, 2n)
        u_f = torch.fft.fft(u) # (batch_size, 2n)
        

        circulant_product = torch.fft.ifft(H_f * u_f[:, np.newaxis])
        if is_complex:
            return circulant_product[..., :n]
        else:
            return circulant_product[..., :n].abs()
        
    # 2D case
    elif dim == 2:
        
        # zero-pad H and u to (2n, 2n)
        
        # Width to 2n
        H = torch.cat((H, torch.zeros_like(H)), dim=-1) # (rank, n, 2n)
        u = torch.cat((u, torch.zeros_like(u)), dim=-1) # (batch_size, n, 2n)
        
        # Height to 2n
        H = torch.cat((H, torch.zeros_like(H)), dim=-2) # (rank, 2n, 2n)
        u = torch.cat((u, torch.zeros_like(u)), dim=-2) # (batch_size, 2n, 2n)
        
        H_f = torch.fft.fft2(H) # (rank, 2n, 2n)
        u_f = torch.fft.fft2(u) # (batch_size, 2n, 2n)
        
        circulant_product = torch.fft.ifft2(H_f * u_f[:, np.newaxis]) # (rank, 2n, 2n)
        
        if is_complex:
            return circulant_product[..., :n, :n]
        else:
            return circulant_product[..., :n, :n].abs()
        
        
        
    
def toeplitz_multiply_fft(G, w, dim = 1, is_complex=False):
    """Multiply SUM_i L(G_i) @ w_i (where L is the lower-triangular Toeplitz matrix) using FFT.
    Parameters:
        G: 1D: (rank, n) or 2D: (rank, n, n)
        w: 1D: (batch_size, rank, n) or 2D: (batch_size, rank, n, n)
        dim: whether to perform 1D or 2D multiplication
    Returns:
        product: (batch, n)
    """
    n = G.shape[1]
    
    # 1D case
    if dim == 1:
        # zero-pad G and w to 2n
        G = torch.cat((G, torch.zeros_like(G)), dim=-1) # (rank, 2n)
        w = torch.cat((w, torch.zeros_like(w)), dim=-1) # (batch_size, rank, 2n)
        
        # shift G
        G_shifted = torch.zeros_like(G)
        G_shifted[:, 0] = G[:, 0]
        G_shifted[:, n+1:] = torch.flip(G[:, 1:n], dims=(-1,))
        
        G_f = torch.fft.fft(G_shifted)
        w_f = torch.fft.fft(w)
        circulant_product = torch.fft.ifft(G_f * w_f)
        if is_complex:
            return circulant_product[..., :n].sum(dim=1)
        else:
            return circulant_product[..., :n].sum(dim=1).abs()
        
    # 2D case
    elif dim == 2:
        
        # zero-pad G and w to (2n, 2n)
        G = torch.cat((G, torch.zeros_like(G)), dim=-1) # (rank, n, 2n)
        w = torch.cat((w, torch.zeros_like(w)), dim=-1) # (batch_size, rank, n, 2n)
        
        G = torch.cat((G, torch.zeros_like(G)), dim=-2) # (rank, 2n, 2n)
        w = torch.cat((w, torch.zeros_like(w)), dim=-2) # (batch_size, rank, 2n, 2n)
        
        # shift G
        G_shifted = torch.zeros_like(G)
        G_shifted[:, 0, 0:n] = G[:, 0, 0:n] # keep first row
        G_shifted[:, 1:n, 0] = G[:, 1:n, 0] # keep first column
        
        G_shifted[:, n+1:, n+1:] = torch.flip(G[:, 1:n, 1:n], dims=(-1,-2)) # fill lower right corner with reverse order of G
        
        G_f = torch.fft.fft2(G_shifted) # (rank, 2n, 2n)
        w_f = torch.fft.fft2(w) # (batch_size, rank, 2n, 2n)
        
        circulant_product = torch.fft.ifft2(G_f * w_f) # (batch_size, rank, 2n, 2n)
        
        if is_complex:
            return circulant_product[..., :n, :n].sum(dim=1)
        else:
            return circulant_product[..., :n, :n].sum(dim=1).abs()
        
        


def toeplitz_mult(G, H, x, cycle=True):
    """Multiply \sum_i Krylov(Z_f, G_i) @ Krylov(Z_f, H_i) @ x.
    Parameters:
        G: Tensor of shape (rank, n)
        H: Tensor of shape (rank, n)
        x: Tensor of shape (batch_size, n)
        cycle: whether to use f = (1, -1) or f = (0, 0)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    # f = (1,-1) if cycle else (1,1)
    f = (1, -1) if cycle else (0, 0)
    transpose_out = toeplitz_krylov_transpose_multiply(H, x, f[1])
    return toeplitz_krylov_multiply(G, transpose_out, f[0])

def toeplitz_mult_symmetric(G, H, x, dim=1, cycle=False, is_complex=False):
    """Multiply \sum_i Krylov(Z_f, G_i) @ Krylov(Z_f, H_i) @ x.
    Parameters:
        G: Tensor of shape (rank, n)
        x: Tensor of shape (batch_size, n)
        cycle: whether to use f = (1, -1) or f = (0, 0)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    
    if cycle:
        raise NotImplementedError("Symmetric toeplitz multiplication with cycle not implemented")
    # f = (1,-1) if cycle else (1,1)
    transpose_out = toeplitz_transpose_multiply_fft(H, x, dim = dim, is_complex=is_complex)
    return toeplitz_multiply_fft(G, transpose_out, dim = dim, is_complex=is_complex)


##### Slow multiplication for the Toeplitz-like case

def toeplitz_Z_f_linear_map(f=0.0):
    """The linear map for multiplying by Z_f.
    This implementation is slow and not batched wrt rank, but easy to understand.
    Parameters:
        f: real number
    Returns:
        linear_map: v -> product, with v of shape (n, )
    """
    return lambda v: torch.cat((f * v[[-1]], v[:-1]))


def krylov_toeplitz_fast(v, f=0.0):
    """Explicit construction of Krylov matrix [v  A @ v  A^2 @ v  ...  A^{n-1} @ v]
    where A = Z_f. This uses vectorized indexing and cumprod so it's much
    faster than using the Krylov function.
    Parameters:
        v: the starting vector of size n or (rank, n).
        f: real number
    Returns:
        K: Krylov matrix of size (n, n) or (rank, n, n).
    """
    rank, n  = v.shape
    a = torch.arange(n, device=v.device)
    b = -a
    indices = a[:, np.newaxis] + b[np.newaxis]
    K = v[:, indices]
    K[:, indices < 0] *= f
    return K


def toeplitz_mult_slow(G, H, x, cycle=True):
    """Multiply \sum_i Krylov(Z_f, G_i) @ Krylov(Z_f, H_i) @ x.
    Uses the explicit Krylov construction with slow (and easy to understand)
    linear map.
    Parameters:
        G: Tensor of shape (rank, n)
        H: Tensor of shape (rank, n)
        x: Tensor of shape (batch_size, n)
        cycle: whether to use f = (1, -1) or f = (0, 0)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    assert G.shape == H.shape, 'G and H must have the same shape'
    rank, n = G.shape
    f = (1, -1) if cycle else (0, 0)
    krylovs = [(Krylov(toeplitz_Z_f_linear_map(f[0]), G[i]), Krylov(toeplitz_Z_f_linear_map(f[1]), H[i]).t()) for i in range(rank)]
    prods = [K[0] @ (K[1] @ x.t()) for K in krylovs]
    return sum(prods).t()


def toeplitz_mult_slow_fast(G, H, x, cycle=True):
    """Multiply \sum_i Krylov(Z_f, G_i) @ Krylov(Z_f, H_i) @ x.
    Uses the fast construction of Krylov matrix.
    Parameters:
        G: Tensor of shape (rank, n)
        H: Tensor of shape (rank, n)
        x: Tensor of shape (batch_size, n)
        cycle: whether to use f = (1, -1) or f = (0, 0)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    assert G.shape == H.shape
    f_G, f_H = (1, -1) if cycle else (0, 0)
    K_G, K_H = krylov_toeplitz_fast(G, f_G), krylov_toeplitz_fast(H, f_H)
    return ((x @ K_H) @ K_G.transpose(1, 2)).sum(dim=0)


def test_toeplitz_mult():
    v = torch.tensor([[0,1,0,-1],[0,1,2,3]], dtype=torch.float, device=device, requires_grad=True)
    u = torch.tensor([[1,1,1,1],[0,1,2,3]], dtype=torch.float, device=device, requires_grad=True)

    w = toeplitz_krylov_transpose_multiply(v, u, f=-1)
    # output:
    # [[[ 0 2  2 0]
    #   [ 6 0 -4 -6]]

    #  [[ -2 2 4  2]
    #   [ 14 8 0 -8]]]

    toeplitz_mult(v, v, u)
    toeplitz_mult_slow(v, v, u)
    # output:
    # array([[-16., -20.,  -4.,  16.],
    #        [ 16.,  -8.,  12.,  64.]])

    toeplitz_mult(v, v, u, cycle=False)
    toeplitz_mult_slow(v, v, u, cycle=False)
    # output:
    # array([[ 0.,  6., 16., 26.],
    #        [ 0., 12., 38., 66.]])

    m = 10
    n = 1<<m
    batch_size = 50
    rank = 16
    u = torch.rand((batch_size, n), requires_grad=True, device=device)
    v = torch.rand((rank, n), requires_grad=True, device=device)
    result = toeplitz_mult(v, v, u, cycle=True)
    grad, = torch.autograd.grad(result.sum(), v, retain_graph=True)
    result_slow = toeplitz_mult_slow(v, v, u, cycle=True)
    grad_slow, = torch.autograd.grad(result_slow.sum(), v, retain_graph=True)
    result_slow_fast = toeplitz_mult_slow_fast(v, v, u, cycle=True)
    grad_slow_fast, = torch.autograd.grad(result_slow_fast.sum(), v, retain_graph=True)
    # These max and mean errors should be small
    print((result - result_slow).abs().max().item())
    print((result - result_slow).abs().mean().item())
    print((grad - grad_slow).abs().max().item())
    print((grad - grad_slow).abs().mean().item())
    print((result - result_slow_fast).abs().max().item())
    print((result - result_slow_fast).abs().mean().item())
    print((grad - grad_slow_fast).abs().max().item())
    print((grad - grad_slow_fast).abs().mean().item())


def test_memory():
    """Memory stress test to make sure there's no memory leak.
    """
    for _ in range(10000):
        a = torch.empty((2,4096), dtype=torch.float, device=device, requires_grad=True)
        b = torch.empty((2,4096), dtype=torch.float, device=device, requires_grad=True)
        c = toeplitz_mult(a, a, b)
        g, = torch.autograd.grad(torch.sum(c), a, retain_graph=True)


# TODO: move test into subpackage
if __name__ == '__main__':
    test_toeplitz_mult()
    # test_memory()
