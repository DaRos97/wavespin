""" Here I compute the vertices to check what's going on.
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openHamiltonian
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from scipy.fft import dctn, fft2,fftshift,fftfreq,dstn

parameters = importParameters()
parameters.dia_plotWf = False
parameters.dia_saveWf = True
parameters.sca_saveVertex = True
parameters.sca_saveRate = False

inp = sys.argv[1]

if inp[0]=='0':       # Compute bands in OBC and PBC
    Lx = 10
    Ly = 12
    Ns = Lx*Ly
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij')
    J1 = 1
    J2 = 0.4
    h = 0
    parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)

    # OBC and PBC
    parameters.lat_boundary = 'open'
    mySystemO = openHamiltonian(parameters)
    parameters.lat_boundary = 'periodic'
    mySystemP = openHamiltonian(parameters)
    #
    evs = mySystemP.evals
    dis = mySystemP.dispersion

    if inp[1]=='a':   # PLot eigenvectors and dctn and fft2
        ni = 0
        nf = 10
        N = nf-ni
        fig = plt.figure(figsize=(2*N,10))
        for i in range(N):
            ax = fig.add_subplot(4,N,i+1,projection='3d')
            ax.plot_surface(X,Y,mySystemO.Phi[:,ni+i].reshape(Lx,Ly),cmap='plasma')
            ax.set_title("E:%.3f"%mySystemO.evals[ni+i])
            #
            ax = fig.add_subplot(4,N,i+N+1,projection='3d')
            dct = dctn(
                mySystemO.Phi[:,ni+i].reshape(Lx,Ly),
                type=2
            )
            ax.plot_surface(X,Y,np.absolute(dct),cmap='inferno')
            #
            ax = fig.add_subplot(4,N,i+2*N+1,projection='3d')
            ax.plot_surface(X,Y,mySystemP.Phi[:,ni+i].reshape(Lx,Ly),cmap='plasma')
            ax.set_title("E:%.3f"%mySystemP.evals[ni+i])
            #
            ax = fig.add_subplot(4,N,i+3*N+1,projection='3d')
            fft = fft2(
                mySystemP.Phi[:,ni+i].reshape(Lx,Ly),
                #type=2
            )
            ax.plot_surface(X,Y,np.absolute(fft),cmap='inferno')
        plt.suptitle("%dx%d lattice"%(Lx,Ly),size=30)
        fig.tight_layout()
        plt.show()
    if inp[1]=='b':
        for i in range(Ns):
            print("Index %d, energy %.5f"%(i,evs[i]))
            indices = np.argwhere(np.absolute(dis-evs[i])<1e-8)     #list of (kx,ky)
            fft = np.absolute(fft2(mySystemP.Phi[:,i].reshape(Lx,Ly)))
            kx,ky = np.unravel_index(np.argmax(fft),fft.shape)

            print(kx,ky)
            print(indices-4)
            input()
if inp[0]=='1':           # Compute vertices in nice plot, OBC and PBC
    Lx = 6
    Ly = 6
    Ns = Lx*Ly
    X,Y = np.meshgrid(np.arange(Ns),np.arange(Ns),indexing='ij')
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    J1 = 1
    J2 = 0
    h = 0
    parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)

    # OBC and PBC
    parameters.lat_boundary = 'periodic'
    mySystemP = openHamiltonian(parameters)
    parameters.lat_boundary = 'open'
    mySystemO = openHamiltonian(parameters)

    resP = {}
    resO = {}
    vertices = ('1to2','2to2','1to3')
    for vertex in vertices:
        print(vertex)
        mySystemP.computeVertex(vertex)
        resP[vertex] = abs(getattr(mySystemP,'vertex'+vertex))
        mySystemO.computeVertex(vertex)
        resO[vertex] = abs(getattr(mySystemO,'vertex'+vertex))

    if inp[1]=='a':   # 1->2 vertex, PBC and OBC, first N modes
        vertex = '1to2'
        cmap = plt.cm.bwr
        ni = 25
        nf = 36
        N = nf-ni
        fig,axs = plt.subplots(2,N+1,figsize=(N*2-3,4),width_ratios=list(np.ones(N))+[0.2,])
        normP = Normalize(vmin=np.amin(resP[vertex][ni:nf,:,:]),vmax=np.amax(resP[vertex][ni:nf,:,:]))
        normO = Normalize(vmin=np.amin(resO[vertex][ni:nf,:,:]),vmax=np.amax(resO[vertex][ni:nf,:,:]))
        for n in range(N):
            axs[0,n].set_title("Index %d"%(ni+n),size=15)
            pmP = axs[0,n].pcolormesh(
                X,Y,
                resP[vertex][ni+n],
                cmap=cmap,
                norm=normP
            )

            pmO = axs[1,n].pcolormesh(
                X,Y,
                resO[vertex][ni+n],
                cmap=cmap,
                norm=normO
            )
            for i in range(2):
                axs[i,n].set_aspect('equal')
                axs[i,n].set_xticks([])
                axs[i,n].set_yticks([])

        # Colorbars
        ax = axs[0,-1]
        sm = mpl.cm.ScalarMappable(norm=normP, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax)
        cbar.set_label("PBC",fontsize=15)

        ax = axs[1,-1]
        sm = mpl.cm.ScalarMappable(norm=normO, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax)
        cbar.set_label("OBC",fontsize=15)

        fig.tight_layout()
        plt.show()
    if inp[1]=='b':       # 2->2 and 1->3 vertex, OBC
        vertex = '1to3'
        indI = 2

        # scattering rates
        parameters.sca_types = (vertex+'_1',)
        parameters.sca_temperature = 0 if vertex=='1to3' else 1
        parameters.sca_broadening = 1e-5
        mySystemO = openHamiltonian(parameters)
        mySystemO.computeRate()
        ratesO = mySystemO.rates[vertex+'_1']

        # figure rates
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(121)
        ax.scatter(mySystemO.evals[1:],ratesO,color='b')
        ax.scatter(mySystemO.evals[indI],ratesO[indI-1],color='r')
        ax.set_xlabel('Energy',size=15)
        ax.set_ylabel("Vertex "+vertex,size=15)
        ax = fig.add_subplot(122)
        ax.scatter(np.arange(1,Ns),ratesO,color='b')
        ax.scatter(indI,ratesO[indI-1],color='r')
        ax.set_xlabel('Mode index',size=15)
        ax.set_ylabel("Vertex "+vertex,size=15)
        plt.suptitle("Lattice %dx%d"%(Lx,Ly),size=20)
        fig.tight_layout()

        # figure vertex
        cmap = plt.cm.bwr
        X,Y = np.meshgrid(np.arange(Ns),np.arange(Ns),indexing='ij')
        fig,axs = plt.subplots(Lx,Ly,figsize=(Lx*2,Ly*2))
        normO = Normalize(vmin=np.amin(resO[vertex][indI,:,:,:]),vmax=np.amax(resO[vertex][indI,:,:,:]))

        evals = mySystemO.evals
        gamma = 5 * mySystemO.p.sca_broadening * np.mean(evals[2:] - evals[1:-1])
        for j in range(Ns):
            ax = axs[j//Ly,j%Ly]
            ax.set_title("index j=%d"%j)
            pmO = ax.pcolormesh(
                X,Y,
                resO[vertex][indI,j],
                cmap=cmap,
                norm=normO
            )
            # Energy window
            Ej = evals[j]
            if vertex == '1to3':
                Ej = -evals[j]
            Emax = evals[indI] + Ej + gamma
            Emin = evals[indI] + Ej - gamma
            if j and Emax>0:
                minLv = []
                maxLv = []
                for k in range(1,Ns):
                    minL = np.argmax(evals + evals[k] - Emin > 0)
                    maxL = np.argmax(~(evals + evals[k] - Emax < 0))
                    if minL > 0 and minL < Ns-1:
                        minLv.append((k,minL))
                    if maxL > 0:# and maxL < Ns-1:
                        maxLv.append((k,maxL))
                minLv = np.array(minLv)
                maxLv = np.array(maxLv)
                s = 0.7
                if minLv.any():
                    ax.plot(minLv[:,0],minLv[:,1],color='y',lw=s)
                if maxLv.any():
                    ax.plot(maxLv[:,0],maxLv[:,1],color='y',lw=s)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle("Vertex %s, index i=%d"%(vertex,indI),fontsize=15)
        plt.tight_layout()
        plt.show()
if inp[0]=='2':           # Compute vertices frustrated branch in OBC
    Lx = 10
    Ly = 10
    Ns = Lx*Ly
    X,Y = np.meshgrid(np.arange(Ns),np.arange(Ns),indexing='ij')
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    J1 = 1
    J2 = 0.
    h = 0
    parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)

    # scattering and boundary
    parameters.lat_boundary = 'open'
    parameters.sca_types = ('2to2_1',)
    parameters.sca_temperature = 1
    parameters.sca_broadening = 0.5

    # Open system
    mySystemO = openHamiltonian(parameters)
    if inp[1]=='a':     # Plot rates and vertex
        # Rate
        mySystemO.computeRate()
        verO = abs(getattr(mySystemO,'vertex2to2'))
        ratesO = mySystemO.rates['2to2_1']

        # Index to plot
        indI = 45

        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(121)
        ax.scatter(mySystemO.evals[1:],ratesO,color='b')
        ax.scatter(mySystemO.evals[indI],ratesO[indI-1],color='r')
        ax.set_xlabel('Energy',size=15)
        ax.set_ylabel("Vertex 2to2",size=15)
        ax = fig.add_subplot(122)
        ax.scatter(np.arange(1,Ns),ratesO,color='b')
        ax.scatter(indI,ratesO[indI-1],color='r')
        ax.set_xlabel('Mode index',size=15)
        ax.set_ylabel("Vertex 2to2",size=15)
        plt.suptitle("Lattice %dx%d"%(Lx,Ly),size=20)
        fig.tight_layout()

        # figure vertex
        cmap = plt.cm.bwr
        X,Y = np.meshgrid(np.arange(Ns),np.arange(Ns),indexing='ij')
        ni = indI-8
        nf = indI+8
        N = nf-ni
        fig,axs = plt.subplots(4,4,figsize=(10,10))
        normO = Normalize(vmin=np.amin(verO[indI,ni:nf,:,:]),vmax=np.amax(verO[indI,ni:nf,:,:]))

        evals = mySystemO.evals
        gamma = 5 * mySystemO.p.sca_broadening * np.mean(evals[2:] - evals[1:-1])
        for j in range(N):
            ax = axs[j//4,j%4]
            ax.set_title("index j=%d"%(ni+j))
            pmO = ax.pcolormesh(
                X,Y,
                verO[indI,ni+j],
                cmap=cmap,
                norm=normO
            )
            # Energy window
            Ej = evals[ni+j]
            Emax = evals[indI] + Ej + gamma
            Emin = evals[indI] + Ej - gamma
            if j and Emax>0:
                minLv = []
                maxLv = []
                for k in range(1,Ns):
                    minL = np.argmax(evals + evals[k] - Emin > 0)
                    maxL = np.argmax(~(evals + evals[k] - Emax < 0))
                    if minL > 0 and minL < Ns-1:
                        minLv.append((k,minL))
                    if maxL > 0:# and maxL < Ns-1:
                        maxLv.append((k,maxL))
                minLv = np.array(minLv)
                maxLv = np.array(maxLv)
                s = 0.7
                if minLv.any():
                    ax.plot(minLv[:,0],minLv[:,1],color='y',lw=s)
                if maxLv.any():
                    ax.plot(maxLv[:,0],maxLv[:,1],color='y',lw=s)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle("Vertex 2to2, index i=%d"%indI,fontsize=15)
        plt.tight_layout()
        plt.show()
    if inp[1]=='b':     # Plot band in OBC
        if J2==0.4:# Specific for 10x10, J2=0.4
            inds = np.array([
                34,35,30,31,26,27,21,22,
                42,43,45,46,38,39,32,33
            ])
        else:
            inds = np.array([])
        en = np.zeros((Lx,Ly))
        pts = []
        from wavespin.static.momentumTransformation import extractMomentum
        for i in range(1,Ns):
            kx,ky = extractMomentum(mySystemO.Phi[:,i].reshape(Lx,Ly))
            en[kx,ky] = mySystemO.evals[i]
            pts.append((i,kx/Lx*np.pi,ky/Ly*np.pi))
        en[en==0] = np.nan
        en[0,0] = 0

        # Figure BZ
        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(121)
        pts = np.array(pts)
        ax.scatter(
            pts[:,1],
            pts[:,2],
            color='r',
            alpha=0.3,
            lw=0,
            s=100
        )
        for i in range(pts.shape[0]):
            ax.text(
                pts[i,1]-0.1,
                pts[i,2]+0.05,
                "ind:%d"%int(pts[i,0]),
                size=10
            )
        for l in inds:
            i = np.argmax(pts[:,0]==l)
            ax.scatter(
                pts[i,1],
                pts[i,2],
                edgecolor='b',
                facecolor='none',
                lw=3,
                s=100
            )
        ax.set_aspect('equal')
        ax.set_title("BZ occupation")
        ax.set_xlabel('Kx')
        ax.set_ylabel('Ky')

        ax = fig.add_subplot(122,projection='3d')
        parameters.lat_boundary = 'periodic'
        X,Y = np.meshgrid(np.arange(1,Lx+1)/Lx*2*np.pi - np.pi,np.arange(1,Ly+1)/Ly*2*np.pi - np.pi,indexing='ij')
        mySystemP = openHamiltonian(parameters)
        ax.plot_surface(
            X,Y,
            mySystemP.dispersion,
            cmap='viridis',
            alpha=0.3
        )
        X,Y = np.meshgrid(np.arange(Lx)/Lx*np.pi,np.arange(Ly)/Ly*np.pi,indexing='ij')
        ax.scatter(
            X,Y,
            en,
            color='r'
        )
        for l in inds:
            i = np.argmax(pts[:,0]==l)
            ax.scatter(
                pts[i,1],
                pts[i,2],
                mySystemO.evals[int(pts[i,0])],
                edgecolor='b',
                facecolor='none',
                lw=3,
                s=100
            )
        ax.view_init(elev=30,azim=-80)

        plt.suptitle(r"$J_2=$%.3f"%J2,size=30)
        plt.tight_layout()

        # Figure Rates
        mySystemO.computeRate()
        verO = abs(getattr(mySystemO,'vertex2to2'))
        ratesO = mySystemO.rates['2to2_1']


        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(121)
        ax.scatter(mySystemO.evals[1:],ratesO,color='r',s=50)
        ax.scatter(
            mySystemO.evals[inds],
            ratesO[inds-1],
            facecolor='none',
            edgecolor='b',
            s=50,
            lw=2
        )
        ax.set_xlabel('Energy',size=15)
        ax.set_ylabel("Vertex 2to2",size=15)
        ax = fig.add_subplot(122)
        ax.scatter(np.arange(1,Ns),ratesO,color='r',s=50)
        ax.scatter(inds,ratesO[inds-1],facecolor='none',edgecolor='b',s=50,lw=2)
        ax.set_xlabel('Mode index',size=15)
        ax.set_ylabel("Vertex 2to2",size=15)
        plt.suptitle("Lattice %dx%d"%(Lx,Ly),size=20)
        fig.tight_layout()


        plt.show()
if inp[0]=='3':           # Compute vertices critical branch in OBC
    Lx = 8
    Ly = 10
    Ns = Lx*Ly
    X,Y = np.meshgrid(np.arange(Ns),np.arange(Ns),indexing='ij')
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    J1 = 1
    J2 = 0
    h = 0.7
    parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)

    # scattering and boundary
    parameters.lat_boundary = 'open'
    parameters.sca_types = ('2to2_1',)
    parameters.sca_temperature = 1
    parameters.sca_broadening = 0.5

    # Open system
    mySystemO = openHamiltonian(parameters)

    if inp[1]=='a':     # Plot rates and vertex
        # Rate
        mySystemO.computeRate()
        verO = abs(getattr(mySystemO,'vertex2to2'))
        ratesO = mySystemO.rates['2to2_1']

        # Index to plot
        indI = 45

        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(121)
        ax.scatter(mySystemO.evals[1:],ratesO,color='b')
        ax.scatter(mySystemO.evals[indI],ratesO[indI-1],color='r')
        ax.set_xlabel('Energy',size=15)
        ax.set_ylabel("Vertex 2to2",size=15)
        ax = fig.add_subplot(122)
        ax.scatter(np.arange(1,Ns),ratesO,color='b')
        ax.scatter(indI,ratesO[indI-1],color='r')
        ax.set_xlabel('Mode index',size=15)
        ax.set_ylabel("Vertex 2to2",size=15)
        plt.suptitle("Lattice %dx%d"%(Lx,Ly),size=20)
        fig.tight_layout()

        # figure vertex
        cmap = plt.cm.bwr
        X,Y = np.meshgrid(np.arange(Ns),np.arange(Ns),indexing='ij')
        ni = indI-8
        nf = indI+8
        N = nf-ni
        fig,axs = plt.subplots(4,4,figsize=(10,10))
        normO = Normalize(vmin=np.amin(verO[indI,ni:nf,:,:]),vmax=np.amax(verO[indI,ni:nf,:,:]))

        evals = mySystemO.evals
        gamma = 5 * mySystemO.p.sca_broadening * np.mean(evals[2:] - evals[1:-1])
        for j in range(N):
            ax = axs[j//4,j%4]
            ax.set_title("index j=%d"%(ni+j))
            pmO = ax.pcolormesh(
                X,Y,
                verO[indI,ni+j],
                cmap=cmap,
                norm=normO
            )
            # Energy window
            Ej = evals[ni+j]
            Emax = evals[indI] + Ej + gamma
            Emin = evals[indI] + Ej - gamma
            if j and Emax>0:
                minLv = []
                maxLv = []
                for k in range(1,Ns):
                    minL = np.argmax(evals + evals[k] - Emin > 0)
                    maxL = np.argmax(~(evals + evals[k] - Emax < 0))
                    if minL > 0 and minL < Ns-1:
                        minLv.append((k,minL))
                    if maxL > 0:# and maxL < Ns-1:
                        maxLv.append((k,maxL))
                minLv = np.array(minLv)
                maxLv = np.array(maxLv)
                s = 0.7
                if minLv.any():
                    ax.plot(minLv[:,0],minLv[:,1],color='y',lw=s)
                if maxLv.any():
                    ax.plot(maxLv[:,0],maxLv[:,1],color='y',lw=s)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle("Vertex 2to2, index i=%d"%indI,fontsize=15)
        plt.tight_layout()
        plt.show()
    if inp[1]=='b':     # Plot band in OBC
        # Specific for 8x10
        inds = np.array([
            45,44,43,42,40,39,37,36,32,31,29,28
        ])
        inds0 = np.array([
            1,2,3,4
        ])
        en = np.zeros((Lx,Ly))
        pts = [(0,0,0)]
        from wavespin.static.momentumTransformation import extractMomentum
        for i in range(1,Ns):
            kx,ky = extractMomentum(mySystemO.Phi[:,i].reshape(Lx,Ly))
            en[kx,ky] = mySystemO.evals[i]
            pts.append((i,kx/Lx*np.pi,ky/Ly*np.pi))
        en[en==0] = np.nan
        en[0,0] = 0

        # Figure BZ
        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(121)
        pts = np.array(pts)
        ax.scatter(
            pts[:,1],
            pts[:,2],
            color='r',
            alpha=0.3,
            lw=0,
            s=100
        )
        for i in range(pts.shape[0]):
            ax.text(
                pts[i,1]-0.1,
                pts[i,2]+0.05,
                "ind:%d"%int(pts[i,0]),
                size=10
            )
        for l in inds:
            i = np.argmax(pts[:,0]==l)
            ax.scatter(
                pts[i,1],
                pts[i,2],
                edgecolor='b',
                facecolor='none',
                lw=3,
                s=100
            )
        for l in inds0:
            i = np.argmax(pts[:,0]==l)
            ax.scatter(
                pts[i,1],
                pts[i,2],
                edgecolor='y',
                facecolor='none',
                lw=3,
                s=100
            )
        ax.set_aspect('equal')
        ax.set_title("BZ occupation")
        ax.set_xlabel('Kx')
        ax.set_ylabel('Ky')

        ax = fig.add_subplot(122,projection='3d')
        parameters.lat_boundary = 'periodic'
        X,Y = np.meshgrid(np.arange(1,Lx+1)/Lx*2*np.pi - np.pi,np.arange(1,Ly+1)/Ly*2*np.pi - np.pi,indexing='ij')
        mySystemP = openHamiltonian(parameters)
        ax.plot_surface(
            X,Y,
            mySystemP.dispersion,
            cmap='viridis',
            alpha=0.3
        )
        X,Y = np.meshgrid(np.arange(Lx)/Lx*np.pi,np.arange(Ly)/Ly*np.pi,indexing='ij')
        ax.scatter(
            X,Y,
            en,
            color='r'
        )
        for l in inds:
            i = np.argmax(pts[:,0]==l)
            ax.scatter(
                pts[i,1],
                pts[i,2],
                mySystemO.evals[int(pts[i,0])],
                edgecolor='b',
                facecolor='none',
                lw=3,
                s=100
            )
        for l in inds0:
            i = np.argmax(pts[:,0]==l)
            ax.scatter(
                pts[i,1],
                pts[i,2],
                mySystemO.evals[int(pts[i,0])],
                edgecolor='y',
                facecolor='none',
                lw=3,
                s=100
            )
        ax.view_init(elev=30,azim=-80)

        plt.suptitle(r"$h=$%.3f"%h,size=30)
        plt.tight_layout()

        # Figure Rates
        mySystemO.computeRate()
        verO = abs(getattr(mySystemO,'vertex2to2'))
        ratesO = mySystemO.rates['2to2_1']


        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(121)
        ax.scatter(mySystemO.evals[1:],ratesO,color='r',s=50)
        ax.scatter(
            mySystemO.evals[inds],
            ratesO[inds-1],
            facecolor='none',
            edgecolor='b',
            s=50,
            lw=2
        )
        ax.scatter(
            mySystemO.evals[inds0],
            ratesO[inds0-1],
            facecolor='none',
            edgecolor='y',
            s=50,
            lw=2
        )
        ax.set_xlabel('Energy',size=15)
        ax.set_ylabel("Vertex 2to2",size=15)
        ax = fig.add_subplot(122)
        ax.scatter(np.arange(1,Ns),ratesO,color='r',s=50)
        ax.scatter(inds,ratesO[inds-1],facecolor='none',edgecolor='b',s=50,lw=2)
        ax.scatter(
            inds0,
            ratesO[inds0-1],
            facecolor='none',
            edgecolor='y',
            s=50,
            lw=2
        )
        ax.set_xlabel('Mode index',size=15)
        ax.set_ylabel("Vertex 2to2",size=15)
        plt.suptitle("Lattice %dx%d"%(Lx,Ly),size=20)
        fig.tight_layout()


        plt.show()



























