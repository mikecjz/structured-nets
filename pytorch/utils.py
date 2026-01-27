import torch
import torch.nn as nn

def mse_loss(pred, true):
    loss_fn = nn.MSELoss()
    mse = loss_fn(pred, true)
    accuracy = torch.FloatTensor([0])

    return mse, accuracy

def mse_loss_complex(pred, true):
    diff = pred - true
    mse = torch.mean(torch.abs(diff))
    accuracy = torch.FloatTensor([0])

    return mse, accuracy
    

def cross_entropy_loss(pred, true):
    loss_fn = nn.CrossEntropyLoss()
    _, true_argmax = torch.max(true, 1)
    cross_entropy = loss_fn(pred, true_argmax)

    _, pred_argmax = torch.max(pred, 1)
    correct_prediction = torch.eq(true_argmax, pred_argmax)
    accuracy = torch.mean(correct_prediction.float())

    return cross_entropy, accuracy

def AhA_cartesian(x, SEs, mask, single_coil, is_complex):
    """
    Compute Cartesian A^H A x, where A is the sensitivity encoding matrix
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input image
    SEs : numpy.ndarray
        Sensitivity encodings
    mask : numpy.ndarray
        K-space sampling mask
    
    Returns:
    --------
    numpy.ndarray
        Result of A^H A x operation
    """
    if not single_coil:
        # Element-wise multiplication of x with sensitivity encodings
        x = torch.unsqueeze(x, dim=-1) * SEs
        mask = torch.unsqueeze(mask, dim=-1)
    else:
        x = x
        
    # Apply inverse FFT shifts and FFT2
    x_shifted = torch.fft.ifftshift(x, dim=(1,2))
    Ax = torch.fft.fft2(x_shifted, dim=(1,2)) * mask
    
    # Apply inverse FFT2 and FFT shifts
    temp = torch.fft.fftshift(torch.fft.ifft2(Ax, dim=(1,2)), dim=(1,2))
    
    if not is_complex:
        temp = torch.abs(temp)
    
    if not single_coil:
        # Sum along the coil dimension (assumed to be axis 2)
        if is_complex:
            AhAx = torch.sum(temp * torch.conj(SEs), dim=-1)
        else:
            AhAx = torch.sum(temp * SEs, dim=-1)
    else:
        AhAx = temp
    
    return AhAx

def AhA_toeplitz(x, SEs, mask, single_coil, is_complex):
    """
    Compute Non Cartesian (Toeplitz) A^H A x, where A is the sensitivity encoding matrix
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input image
    SEs : numpy.ndarray
        Sensitivity encodings
    mask : numpy.ndarray
        K-space sampling mask
    
    Returns:
    --------
    numpy.ndarray
        Result of A^H A x operation
    """
    batch_size = x.shape[0]
    n = x.shape[1]
    x_orig = x
    if not single_coil:
        # Element-wise multiplication of x with sensitivity encodings
        x = torch.unsqueeze(x, dim=-1) * SEs
        mask = torch.unsqueeze(mask, dim=-1)
        
        # Zero pad x to 2n
        x_padded = torch.zeros((batch_size, 2*n, 2*n, x.shape[-1]), dtype=x.dtype, device=x.device)
        x_padded[:, n//2:n//2+n, n//2:n//2+n, :] = x
        x = x_padded
        
    else:
        x_padded = torch.zeros((batch_size, 2*n, 2*n), dtype=x.dtype, device=x.device)
        x_padded[:, n//2:n//2+n, n//2:n//2+n] = x
        x = x_padded
        
    
        
    # Apply inverse FFT shifts and FFT2
    x_shifted = torch.fft.ifftshift(x, dim=(1,2))
    Ax = torch.fft.fft2(x_shifted, dim=(1,2)) * mask
    
    # Apply inverse FFT2 and FFT shifts
    temp = torch.fft.fftshift(torch.fft.ifft2(Ax, dim=(1,2)), dim=(1,2))
    
    temp = temp[:, n//2:n//2+n, n//2:n//2+n, ...]
    
    if not is_complex:
        temp = torch.abs(temp)
    
    if not single_coil:
        # Sum along the coil dimension (assumed to be axis 2)
        if is_complex:
            AhAx = torch.sum(temp * torch.conj(SEs), dim=-1)
        else:
            AhAx = torch.sum(temp * SEs, axis=-1)
    else:
        AhAx = temp
    
    return AhAx 


def get_commit_id():
  return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])

def descendants(cls):
    """
    Get all subclasses (recursively) of class cls, not including itself
    Assumes no multiple inheritance
    """
    desc = []
    for subcls in cls.__subclasses__():
        desc.append(subcls)
        desc.extend(descendants(subcls))
    return desc
