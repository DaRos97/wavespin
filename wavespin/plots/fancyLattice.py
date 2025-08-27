""" My version of plotting a lattice.
"""
import matplotlib.pyplot as plt

def plotSitesGrid(openSystem,**kwargs):
    """ Here we plot the grid structure to see which sites are considered in the calculation.

    Parameters
    ----------
    openSystem : openSystem object.
    """
    Lx = openSystem.Lx
    Ly = openSystem.Ly
    offSiteList = openSystem.offSiteList
    perturbationSite = openSystem.perturbationSite
    indexesMap = openSystem.indexesMap
    printIndices = kwargs.get('indices',True)
    #
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot()
    for ix in range(Lx):
        for iy in range(Ly):
            if (ix,iy) in offSiteList:
                ax.scatter(ix,iy,c='r',marker='x',s=80,zorder=2)
            else:
                ax.scatter(ix,iy,c='b',marker='o',s=80,zorder=2)
                if printIndices:
                    ax.text(ix+0.05,iy+0.15,str(indexesMap.index((ix,iy))),size=20)
            if ix+1<Lx:
                if (ix,iy) in offSiteList or (ix+1,iy) in offSiteList:
                    ax.plot([ix,ix+1],[iy,iy],c='y',ls='--',lw=0.5,zorder=-1)
                else:
                    ax.plot([ix,ix+1],[iy,iy],c='darkgreen',ls='-',lw=2,zorder=-1)
            if iy+1<Ly:
                if (ix,iy) in offSiteList or (ix,iy+1) in offSiteList:
                    ax.plot([ix,ix],[iy,iy+1],c='y',ls='--',lw=0.5,zorder=-1)
                else:
                    ax.plot([ix,ix],[iy,iy+1],c='darkgreen',lw=2,zorder=-1)
    ax.scatter(perturbationSite[0],perturbationSite[1],c='w',edgecolor='m',lw=2,marker='o',s=200,zorder=1)
    ax.set_aspect('equal')
    ax.set_xlabel('x',size=30)
    ax.set_ylabel('y',size=30)
    fig.tight_layout()
    plt.show()
