""" Functions for the Fourier transform with different boundary conditions.
"""

import numpy as np
from scipy.fft import fftshift, fft, fft2, dstn, dctn, fftfreq

def DCTgeom(system):
    """ DCT of tilted geometry.
    We separate the lattice in the two sublattices which must have tilted-rectangular shapes.
    We DCT them separately but combining them through the displacement vector.
    Explicitly for 97 qubit geom
    """
    func = system.correlatorXT
    Nr,Mr,Nb,Mb = (8,6,7,7)        #97-geometry
    xr0, xb0 = (5,6)
    yr0, yb0 = (0,0)
    #Nr,Mr,Nb,Mb = (7,6,7,6)         #84 geometry
    #xr0, xb0 = (5,5)
    #yr0, yb0 = (0,1)
    #Separate
    fr = np.zeros((Nr,Mr,system.nTimes),dtype=complex)
    fb = np.zeros((Nb,Mb,system.nTimes),dtype=complex)
    R = np.array([[1,1],[-1,1]]) / 2
    for ind in range(system.Ns):
        x,y = system._xy(ind)
        if (x+y)%2==1:  #sublattice-0
            x -= xr0
            y -= yr0
            x_new, y_new = R@np.array([x,y])
            fr[int(x_new),int(y_new)] = func[ind]
        else:  #sublattice-1
            x -= xb0
            y -= yb0
            x_new, y_new = R@np.array([x,y])
            fb[int(x_new),int(y_new)] = func[ind]
    #Transform dct
    frk = np.zeros_like(fr)
    fbk = np.zeros_like(fb)
    #
    Kxr = np.pi/Nr*(np.arange(Nr) + 0)
    Kyr = np.pi/Mb*(np.arange(Mr) + 1)
    Kxb = np.pi/Nr*(np.arange(Nb) + 1)
    Kyb = np.pi/Mb*(np.arange(Mb) + 0)
    #
    Xr = np.arange(Nr)
    Yr = np.arange(Mr) + 1/2
    Xb = np.arange(Nb) + 1/2
    Yb = np.arange(Mb)
    for it in range(system.nTimes):
        for ikxr in range(Nr):
            for ikyr in range(Mr):
                frk[ikxr,ikyr,it] = np.sum(
                    fr[:,:,it] * np.cos(Kxr[ikxr]*(Xr[:,None]+1/2)) * np.cos(Kyr[ikyr]*(Yr[None,:]+1/2)) )
                frk[ikxr,ikyr,it] += np.sum(
                    fb[:,:,it] * np.cos(Kxr[ikxr]*(Xb[:,None]+1/2)) * np.cos(Kyr[ikyr]*(Yb[None,:]+1/2)) )
        for ikxb in range(Nb):
            for ikyb in range(Mb):
                fbk[ikxb,ikyb,it] = np.sum(
                    fr[:,:,it] * np.cos(Kxb[ikxb]*(Xr[:,None]+1/2)) * np.cos(Kyb[ikyb]*(Yr[None,:]+1/2)) )
                fbk[ikxb,ikyb,it] += np.sum(
                    fb[:,:,it] * np.cos(Kxb[ikxb]*(Xb[:,None]+1/2)) * np.cos(Kyb[ikyb]*(Yb[None,:]+1/2)) )
    #Time transform
    frkw = np.zeros((Nr,Mr,system.nOmega),dtype=complex)
    for ikxr in range(Nr):
        for ikyr in range(Mr):
            frkw[ikxr,ikyr] = fftshift(fft( frk[ikxr,ikyr], n=system.nOmega ))
    fbkw = np.zeros((Nb,Mb,system.nOmega),dtype=complex)
    for ikxb in range(Nb):
        for ikyb in range(Mb):
            fbkw[ikxb,ikyb] = fftshift(fft( fbk[ikxb,ikyb], n=system.nOmega ))
    #Patch together
    fFinal = np.zeros((system.Ns,system.nOmega),dtype=complex)
    fFinal[:Nr*Mr] = frkw.reshape(Nr*Mr,system.nOmega)
    fFinal[Nr*Mr:] = fbkw.reshape(Nb*Mb,system.nOmega)
    #Momenta
    mom_r = np.zeros((Nr,Mr,2))
    mom_b = np.zeros((Nb,Mb,2))
    for ikxr in range(Nr):
        for ikyr in range(Mr):
            mom_r[ikxr,ikyr] = np.array([
                Kxr[ikxr],
                Kyr[ikyr]
            ])
    for ikxb in range(Nb):
        for ikyb in range(Mb):
            mom_b[ikxb,ikyb] = np.array([
                Kxb[ikxb],
                Kyb[ikyb]
            ])
    momentum = np.zeros((system.Ns,2))
    momentum[:Nr*Mr] = mom_r.reshape(Nr*Mr,2)
    momentum[Nr*Mr:] = mom_b.reshape(Nb*Mb,2)
    #
    if 1:
        import matplotlib.pyplot as plt
        X_r,Y_r = np.meshgrid(Xr,Yr,indexing='ij')
        X_b,Y_b = np.meshgrid(Xb,Yb,indexing='ij')
        X,Y = np.meshgrid(np.arange(system.Lx),np.arange(system.Ly),indexing='ij')
        fig = plt.figure(figsize=(20,15))
        for i in range(Nr*Mr):
            ax = fig.add_subplot(Mr,Nr,i+1)
            kx = Kxr[i%Nr]
            ky = Kyr[i//Nr]
            fun_r = np.cos(kx*(X_r+1/2)) * np.cos(ky*(Y_r+1/2))
            fun_b = np.cos(kx*(X_b+1/2)) * np.cos(ky*(Y_b+1/2))
            # Back to Ns function
            fun = np.zeros(system.Ns)
            R = np.array([[1,-1],[1,1]])
            for ixr in range(Nr):
                for iyr in range(Mr):
                    v = R @ np.array([ixr,iyr])
                    ind = system._idx(int(v[0])+xr0,int(v[1])+yr0)
                    fun[ind] = fun_r[ixr,iyr]
            for ixb in range(Nb):
                for iyb in range(Mb):
                    v = R @ np.array([ixb,iyb])
                    ind = system._idx(int(v[0])+xb0,int(v[1])+yb0)
                    fun[ind] = fun_b[ixb,iyb]
            # Patch and plot
            fun_patched = system.patchFunction(fun)
            ax.pcolormesh(X,Y,fun_patched,cmap='bwr')
            ax.set_title("kx,ky: %d,%d"%(i//Nr,i%Nr))
            ax.set_aspect('equal')
        plt.suptitle("Red sublattice")
        fig.tight_layout()
        #plt.show()
        fig = plt.figure(figsize=(20,25))
        for i in range(Nb*Mb):
            ax = fig.add_subplot(Nb,Mb,i+1)
            kx = Kxb[i%Nb]
            ky = Kyb[i//Nb]
            fun_r = np.cos(kx*(X_r+1/2)) * np.cos(ky*(Y_r+1/2))
            fun_b = np.cos(kx*(X_b+1/2)) * np.cos(ky*(Y_b+1/2))
            # Back to Ns function
            fun = np.zeros(system.Ns)
            R = np.array([[1,-1],[1,1]])
            for ixr in range(Nr):
                for iyr in range(Mr):
                    v = R @ np.array([ixr,iyr])
                    ind = system._idx(int(v[0])+xr0,int(v[1])+yr0)
                    fun[ind] = fun_r[ixr,iyr]
            for ixb in range(Nb):
                for iyb in range(Mb):
                    v = R @ np.array([ixb,iyb])
                    ind = system._idx(int(v[0])+xb0,int(v[1])+yb0)
                    fun[ind] = fun_b[ixb,iyb]
            # Patch and plot
            fun_patched = system.patchFunction(fun)
            ax.pcolormesh(X,Y,fun_patched,cmap='bwr')
            ax.set_title("kx,ky: %d,%d"%(i//Nb,i%Nb))
            ax.set_aspect('equal')
        plt.suptitle("Blue sublattice")
        fig.tight_layout()
        #plt.show()
        #exit()
    #
    if 0:
        import matplotlib.pyplot as plt
        freqs = fftshift(fftfreq(system.nOmega,system.fullTimeMeasure/system.nTimes))
        for i in range(Nr*Mr):
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot()
            ax.plot(freqs,np.imag(fFinal[i,:]),label=str(i))
            #ax.plot(freqs,np.imag(fFinal[-1,:]),label=str(i)+'t')
            #ax.plot(np.arange(system.nTimes),np.imag(f0k.reshape(N0*M0,system.nTimes)[i,:]),label=str(i))
            #ax.plot(np.arange(system.nTimes),np.imag(f0.reshape(N0*M0,system.nTimes)[i,:]),label=str(i))
            ax.set_xlim(-70,70)
            ax.legend()
            plt.show()
        exit()

    return fFinal, momentum

def DCTgeom2(system):
    nOmega = system.nOmega
    Lx = system.Lx
    Ly = system.Ly
    kx = np.pi * np.arange(0, Lx ) / (Lx )
    ky = np.pi * np.arange(0, Ly ) / (Ly )
    nTimes = system.nTimes
    correlatorKW = np.zeros((Lx*Ly,nOmega),dtype=complex)
    momentum = np.zeros((Lx*Ly,2))
    fKT = np.zeros((Lx,Ly,nTimes),dtype=complex)
    for it in range(nTimes):
        fun = correlatorKW[:,it]
        for ikx in range(Lx):
            for iky in range(Ly):
                for ind in range(system.Ns):
                    indx,indy = system._xy(ind)
                    fKT[ikx,iky,it] += fun[ind] * np.cos(kx[ikx]*(indx+1/2)) * np.cos(ky[iky]*(indy+1/2))
    for ikx in range(Lx):
        for iky in range(Ly):
            correlatorKW[ikx*Ly+iky] = fftshift(fft(fKT[ikx,iky],n=nOmega))
            momentum[ikx*Ly+iky] = np.array([kx[ikx],ky[iky]])
    return correlatorKW, momentum

def discreteCosineTransform(system,dctType=2):
    """
    Compute the Discrete Cos Transform since we have open BC.
    In time we always use fft.
    """
    if len(system.offSiteList)>0:
        return DCTgeom(system)
    nOmega = system.nOmega
    Lx = system.Lx
    Ly = system.Ly
    kx = np.pi * np.arange(0, Lx ) / (Lx )
    ky = np.pi * np.arange(0, Ly ) / (Ly )
    nTimes = system.nTimes
    correlatorKW = np.zeros((system.Ns,nOmega),dtype=complex)
    momentum = np.zeros((system.Ns,2))
    temp = np.zeros((Lx,Ly,nTimes),dtype=complex)
    for it in range(nTimes):
        temp[:,:,it] = dctn(system.correlatorXT[:,it].reshape(Lx,Ly),
                            type=dctType,
                            norm='ortho' )
        continue
        # Explicit calculation
        fun = system.correlatorXT[:,it].reshape(Lx,Ly)
        for ix in range(Lx):
            for iy in range(Ly):
                temp[ix,iy,it] = np.sum(
                    fun*np.cos(kx[ix]*(np.arange(Lx)[:,None]+1/2))*np.cos(ky[iy]*(np.arange(Ly)[None,:]+1/2)))
    for ind in range(system.Ns):
        x,y = system._xy(ind)
        correlatorKW[ind] = fftshift(fft(temp[x,y],n=nOmega))
        momentum[ind] = np.array([kx[x],ky[y]])
    if 0:
        import matplotlib.pyplot as plt
        freqs = fftshift(fftfreq(system.nOmega,system.fullTimeMeasure/system.nTimes))
        for i in range(1):
            i = system.Ns-1
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot()
            ax.plot(freqs,np.imag(correlatorKW[i,:]),label=str(i))
            ax.set_xlim(-70,70)
            ax.legend()
            plt.show()
        exit()
    return correlatorKW, momentum

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
