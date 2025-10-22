""" Functions for the Fourier transform with different boundary conditions.
"""

import numpy as np
from scipy.fft import fftshift, fft, fft2, dstn, dctn

def discreteCosineTransform(system,dctType=2):
    """
    Compute the Discrete Cos Transform since we have open BC.
    In time we always use fft.
    """
    nOmega = system.nOmega
    Lx = system.Lx
    Ly = system.Ly
    nTimes = system.nTimes
    correlatorKW = np.zeros((Lx,Ly,nOmega),dtype=complex)
    temp = np.zeros((Lx,Ly,nTimes),dtype=complex)
    for it in range(nTimes):
        temp[:,:,it] = dctn(system.correlatorXT[:,:,it], type=dctType, norm='ortho')
    for ix in range(Lx):
        for iy in range(Ly):
            correlatorKW[ix,iy] = fftshift(fft(temp[ix,iy],n=nOmega))
    return correlatorKW

def fastFourierTransform(system):
    """
    Compute the standard 2D Fourier transform.
    In time we always use fft.
    """
    nOmega = system.nOmega
    Lx = system.Lx
    Ly = system.Ly
    nTimes = system.nTimes
    correlatorKW = np.zeros((Lx,Ly,nOmega),dtype=complex)
    temp = np.zeros((Lx,Ly,nTimes),dtype=complex)
    for it in range(nTimes):
        temp[:,:,it] = fftshift(fft2(system.correlatorXT[:,:,it],norm='ortho'))
    for ix in range(Lx):
        for iy in range(Ly):
            correlatorKW[ix,iy] = fftshift(fft(temp[ix,iy],n=nOmega))
    return correlatorKW

def discreteSinTransform(system,dstType=1):
    """
    Compute the Discrete Sin Transform since we have open BC.
    In time we always use fft.
    """
    nOmega = system.nOmega
    Lx = system.Lx
    Ly = system.Ly
    nTimes = system.nTimes
    correlatorKW = np.zeros((Lx,Ly,nOmega),dtype=complex)
    temp = np.zeros((Lx,Ly,nTimes),dtype=complex)
    for it in range(nTimes):
        temp[:,:,it] = dstn(system.correlatorXT[:,:,it], type=dstType, norm='ortho')
    for ix in range(Lx):
        for iy in range(Ly):
            correlatorKW[ix,iy] = fftshift(fft(temp[ix,iy],n=nOmega))
    return correlatorKW

def discreteAwsomeTransform(system):
    """ Compute the Discrete Awesome Transform with the Bogoliubov functions.
    NON-RECTANGULAR GEOMETRIES NOT YET IMPLEMENTED.
    In time we always use fft.
    """
    nOmega = system.nOmega
    Lx = system.Lx
    Ly = system.Ly
    nTimes = system.nTimes
    U_ = system.U_
    V_ = system.V_
    correlatorXT = system.correlatorXT.reshape(Lx*Ly,nTimes)
    Ns = Lx*Ly
    correlatorKW = np.zeros((Lx,Ly,nOmega),dtype=complex)
    temp = np.zeros((Lx,Ly,nTimes),dtype=complex)
    phi_ik = np.real(U_ - V_)
    for i in range(Ns):
        ix,iy = system._xy(i)
        phi_ik[i,:] *= 2/np.pi*(-1)**(ix+iy+1)
    for k in range(Ns):
        kx,ky = extractMomentum(phi_ik[:,k].reshape(Lx,Ly))
        temp[kx,ky] = np.sum(phi_ik[:,k,None]*correlatorXT[:,:],axis=0)
        correlatorKW[kx,ky] = fftshift(fft(temp[kx,ky],n=nOmega))
    return correlatorKW

def extractMomentum(f_in,ik=0,dctType=2):
    """ We get the momentum associated with a given Bogoliubov mode.
    We do this by computing the peak of the dctn of the input function (mode) to extract the momentum.
    f_in has shape (Lx,Ly).
    """
    Lx,Ly = f_in.shape
    f_in = f_in.T
    X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly))
    #Renormalize the awesome functions by the cosine
    abs_f = np.absolute(f_in)
    res = np.zeros((Lx,Ly))
    for kx in range(Lx):
        for ky in range(Ly):
            cosx = np.cos(np.pi*kx*(2*X+1)/2/Lx)
            cosy = np.cos(np.pi*ky*(2*Y+1)/2/Ly)
            abs_cosx = np.absolute(cosx)
            abs_cosx[abs_cosx==0] = 1
            abs_cosy = np.absolute(cosy)
            abs_cosy[abs_cosy==0] = 1
            res[kx,ky] = np.sum(f_in*abs_f*cosx/abs_cosx*cosy/abs_cosy)
    ind = np.argmax(np.absolute(res))
    kx,ky = ind//Ly, ind%Ly
    return kx,ky

def discreteAwsomeTransform2(system):
    """ Compute the Discrete Awesome Transform with the Bogoliubov functions.
    NON-RECTANGULAR GEOMETRIES NOT YET IMPLEMENTED.
    In time we always use fft.
    """
    nOmega = system.nOmega
    Lx = system.Lx
    Ly = system.Ly
    nTimes = system.nTimes
    U_ = system.U_
    V_ = system.V_
    correlatorXT = system.correlatorXT.reshape(Lx*Ly,nTimes)
    Ns = Lx*Ly
    correlatorKW = np.zeros((Ns,nOmega),dtype=complex)
    temp = np.zeros(Ns,dtype=complex)
    phi_ik = np.real(U_ - V_)
    for i in range(Ns):
        ix,iy = system._xy(i)
        phi_ik[i,:] *= 2/np.pi*(-1)**(ix+iy+1)
    for n in range(1,Ns):
        temp = np.sum(phi_ik[:,n,None]*correlatorXT[:,:],axis=0)
        correlatorKW[n] = fftshift(fft(temp,n=nOmega))
    return correlatorKW     #shape(Ns,Nomega)

def extractMomentum2(system):
    """ We get the momentum associated with a given Bogoliubov mode.
    We get kx,ky for each mode n by summing the sqrt of the weight of the dctn of the mode.
    """
    Lx = system.Lx
    Ly = system.Ly
    Ns = system.Ns
    U_,V_ = system.U_, system.V_
    phi = np.real(U_ - V_)
    for i in range(Ns):
        ix,iy = system._xy(i)
        phi[i,:] *= 2/np.pi*(-1)**(ix+iy+1)
    vecK = np.zeros((Ns,2))
    X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij')
    for n in range(1,Ns):
        f_in = phi[:,n].reshape(Lx,Ly)
        res = np.absolute(dctn(f_in))
        res[0,0] = 0
        res /= np.sum(res)
        kx = np.sum(res * X)
        ky = np.sum(res * Y)
        vecK[n,0] = kx / Lx * np.pi
        vecK[n,1] = ky / Ly * np.pi
    return vecK

dicTransformType = {'fft':fastFourierTransform,
                    'dct':discreteCosineTransform,
                    'dst':discreteCosineTransform,
                    'dat':discreteAwsomeTransform,
                    'dat2':discreteAwsomeTransform2
                    }
