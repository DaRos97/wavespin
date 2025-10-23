""" Functions computig the open boundary conditions - real space correlators.
"""
import numpy as np

def zzCorrelator(system,ind_i,A,B,G,H):
    """ Compute real space <[Z_i(t),Z_j(0)]> correlator.
    A,B,G and H are Ns,Ns.
    """
    ts = system.ts
    site0 = system.site0
    indexesMap = system.indexToSite
    measurementIndex = indexesMap.index(system._xy(ind_i))
    perturbationIndex = system.perturbationIndex
    S = system.S
    magnonModes = system.p.cor_magnonModes
    #
    ts_i = ts[(site0+indexesMap[measurementIndex][0]+indexesMap[measurementIndex][1])%2]
    ts_j = ts[(site0+indexesMap[perturbationIndex][0]+indexesMap[perturbationIndex][1])%2]
    ts_list = [ts_i, ts_j]
    ZZ = np.zeros(A[0,0].shape,dtype=complex)
    for ops in ['XX','ZZ']:
        original_op = 'ZZ'
        list_terms = computeCombinations(ops,[measurementIndex,perturbationIndex],'t0',S)
        coeff_t = computeCoeffT(ops,original_op,ts_list)
        for t in list_terms:
            ZZ += coeff_t * t[0] * computeContraction(t[1],t[2],t[3],A,B,G,H,magnonModes)
    return 2*1j*np.imag(ZZ)

def zeCorrelator(system,ind_i,A,B,G,H):
    """ Compute real space <[Z_i(t),E_j(0)]> correlator.
    A,B,G and H are Ns,Ns.
    """
    Lx = system.Lx
    Ly = system.Ly
    ts = system.ts
    site0 = system.site0
    indexesMap = system.indexToSite
    measurementIndex = indexesMap.index(system._xy(ind_i))
    perturbationIndex = system.perturbationIndex
    S = system.S
    magnonModes = system.magnonModes
    #
    ts_i = ts[(site0+indexesMap[measurementIndex][0]+indexesMap[measurementIndex][1])%2]
    ts_j = ts[(site0+indexesMap[perturbationIndex][0]+indexesMap[perturbationIndex][1])%2]
    ZE = np.zeros(A[0,0].shape,dtype=complex)
    ind_nn_j = get_nn(perturbationIndex,Lx,Ly)
    for i in system.offSiteList:
        if stsem._idx(i) in ind_nn_j:
            ind_nn_j.remove(system._idx(i))
    for ind_s in ind_nn_j:
        isx,isy = system._xy(ind_s)
        ts_s = ts[(site0+isx+isy)%2]
        ts_list = [ts_i, ts_j, ts_s]
        for ops in ['ZXX','XZX','XXZ','ZZZ','ZYY']:
            original_op = 'ZXX'
            if ops=='ZYY':
                original_op = 'ZYY'
            list_terms = computeCombinations(ops,[ind_i,perturbationIndex,ind_s],'t00',S)
            coeff_t = computeCoeffT(ops,original_op,ts_list)
            for t in list_terms:
                ZE += coeff_t * t[0] * computeContraction(t[1],t[2],t[3],A,B,G,H,magnonModes)
    return 2*1j/len(ind_nn_j)*np.imag(ZE)

def ezCorrelator(system,ind_i,A,B,G,H):
    """ Compute real space <[E_i(t),Z_j(0)]> correlator.
    A,B,G and H are Ns,Ns.
    """
    Lx = system.Lx
    Ly = system.Ly
    ts = system.ts
    site0 = system.site0
    indexesMap = system.indexToSite
    measurementIndex = indexesMap.index(system._xy(ind_i))
    perturbationIndex = system.perturbationIndex
    S = system.S
    magnonModes = system.magnonModes
    #
    ts_i = ts[(site0+indexesMap[measurementIndex][0]+indexesMap[measurementIndex][1])%2]
    ts_j = ts[(site0+indexesMap[perturbationIndex][0]+indexesMap[perturbationIndex][1])%2]
    EZ = np.zeros(A[0,0].shape,dtype=complex)
    ind_nn_i = get_nn(ind_i,Lx,Ly)
    for i in system.offSiteList:
        if system._idx(i) in ind_nn_i:
            ind_nn_i.remove(system._idx(i))
    for ind_r in ind_nn_i:
        irx,iry = system._xy(ind_r)
        ts_r = ts[(site0+irx+iry)%2]
        ts_list = [ts_i, ts_r, ts_j]
        for ops in ['XXZ','XZX','ZXX','ZZZ','YYZ']:
            original_op = 'XXZ'
            if ops=='YYZ':
                original_op = 'YYZ'
            list_terms = computeCombinations(ops,[ind_i,ind_r,perturbationIndex],'tt0',S)
            coeff_t = computeCoeffT(ops,original_op,ts_list)
            for t in list_terms:
                EZ += coeff_t * t[0] * computeContraction(t[1],t[2],t[3],A,B,G,H,magnonModes)
    return 2*1j/len(ind_nn_i)*np.imag(EZ)

def eeCorrelator(system,ind_i,A,B,G,H):
    """ Compute real space <[E_i(t),E_j(0)]> correlator.
    A,B,G and H are Ns,Ns.
    Site j is where the E perturbation is applied -> we assume it is
    somewhere in the middle which has all 4 nearest neighbors and average
    over them.
    """
    Lx = system.Lx
    Ly = system.Ly
    ts = system.ts
    site0 = system.site0
    indexesMap = system.indexToSite
    measurementIndex = indexesMap.index(system._xy(ind_i))
    perturbationIndex = system.perturbationIndex
    S = system.S
    magnonModes = system.p.cor_magnonModes
    #
    ts_i = ts[(site0+indexesMap[measurementIndex][0]+indexesMap[measurementIndex][1])%2]
    ts_j = ts[(site0+indexesMap[perturbationIndex][0]+indexesMap[perturbationIndex][1])%2]
    EE = np.zeros(A[0,0].shape,dtype=complex)
    ind_nn_i = get_nn(ind_i,Lx,Ly)
    ind_nn_j = [perturbationIndex+Ly,]
    Jbond = np.mean(system.g_i[0][ind_i,ind_nn_i])
    for i in system.offSiteList:
        if system._idx(i) in ind_nn_i:
            ind_nn_i.remove(system._idx(i))
        if system._idx(i) in ind_nn_j:
            ind_nn_j.remove(system._idx(i))
    for ind_r in ind_nn_i:
        irx,iry = system._xy(ind_r)
        ts_r = ts[(site0+irx+iry)%2]
        for ind_s in ind_nn_j:
            isx,isy = system._xy(ind_s)
            ts_s = ts[(site0+isx+isy)%2]
            ts_list = [ts_i,ts_r,ts_j,ts_s]
            for ops in ['XXXX','ZZZZ','XXZZ','ZZXX','XZXZ','ZXZX','ZXXZ','XZZX','XXYY','ZZYY','YYXX','YYZZ','YYYY']:
                original_op = 'XXXX'
                if ops in ['XXYY','ZZYY']:
                    original_op = 'XXYY'
                if ops in ['YYXX','YYZZ']:
                    original_op = 'YYXX'
                if ops == 'YYYY':
                    original_op = 'YYYY'
                list_terms = computeCombinations(ops,[ind_i,ind_r,perturbationIndex,ind_s],'tt00',S)
                coeff_t = computeCoeffT(ops,original_op,ts_list)
                for t in list_terms:
                    contraction = computeContraction(t[1],t[2],t[3],A,B,G,H,magnonModes)
                    EE += coeff_t * t[0] * contraction
    return 2*1j/len(ind_nn_i)/len(ind_nn_j)*np.imag(EE) * Jbond**2

def xxCorrelator(system,ind_i,A,B,G,H):
    """ Compute real space <[X_i(t),X_j(0)]> correlator.
    A,B,G and H are Ns,Ns.
    """
    ts = system.ts
    site0 = system.site0
    indexesMap = system.indexToSite
    measurementIndex = indexesMap.index(system._xy(ind_i))
    perturbationIndex = system.perturbationIndex
    S = system.S
    magnonModes = system.magnonModes
    #
    ts_i = ts[(site0+indexesMap[measurementIndex][0]+indexesMap[measurementIndex][1])%2]
    ts_j = ts[(site0+indexesMap[perturbationIndex][0]+indexesMap[perturbationIndex][1])%2]
    ts_list = [ts_i, ts_j]
    XX = np.zeros(A[0,0].shape,dtype=complex)
    for ops in ['XX','ZZ']:
        original_op = 'XX'
        list_terms = computeCombinations(ops,[ind_i,perturbationIndex],'t0',S)
        coeff_t = computeCoeffT(ops,original_op,ts_list)
        for t in list_terms:
            XX += coeff_t * t[0] * computeContraction(t[1],t[2],t[3],A,B,G,H,magnonModes)
    return 2*1j*np.imag(XX)

def jjCorrelator(system,ind_i,A,B,G,H):
    """ Compute real space <[J_i(t),J_j(0)]> correlator.
    A,B,G and H are Ns,Ns.
    Site j is where the J perturbation is applied -> we assume it is
    somewhere in the middle which has all 4 nearest neighbors and average
    over them.
    """
    Lx = system.Lx
    Ly = system.Ly
    ts = system.ts
    site0 = system.site0
    indexesMap = system.indexToSite
    measurementIndex = indexesMap.index(system._xy(ind_i))
    perturbationIndex = system.perturbationIndex
    S = system.S
    magnonModes = system.p.cor_magnonModes
    #
    ts_i = ts[(site0+indexesMap[measurementIndex][0]+indexesMap[measurementIndex][1])%2]
    ts_j = ts[(site0+indexesMap[perturbationIndex][0]+indexesMap[perturbationIndex][1])%2]
    JJ = np.zeros(A[0,0].shape,dtype=complex)
    ind_nn_i = get_nn(ind_i,Lx,Ly)
    Jbond = np.mean(system.g_i[0][ind_i,ind_nn_i])
    ind_nn_j = [perturbationIndex+1,]
    for i in system.offSiteList:
        if system._idx(i) in ind_nn_i:
            ind_nn_i.remove(system._idx(i))
        if system._idx(i) in ind_nn_j:
            ind_nn_j.remove(system._idx(i))
    term_list = ['XYXY','ZYZY','XYYX','ZYYZ','YXXY','YZZY','YXYX','YZYZ']
    for ind_r in ind_nn_i:
        irx,iry = system._xy(ind_r)
        ts_r = ts[(site0+irx+iry)%2]
        for ind_s in ind_nn_j:
            isx,isy = system._xy(ind_s)
            ts_s = ts[(site0+isx+isy)%2]
            ts_list = [ts_i,ts_r,ts_j,ts_s]
            for i,ops in enumerate(term_list):
                original_op = ops if i%2==0 else term_list[i-1]
                list_terms = computeCombinations(ops,[ind_i,ind_r,perturbationIndex,ind_s],'tt00',S)
                coeff_t = computeCoeffT(ops,original_op,ts_list)
                if i in [2,3,4,5]:
                    coeff_t *= -1
                for t in list_terms:
                    contraction = computeContraction(t[1],t[2],t[3],A,B,G,H,magnonModes)
                    JJ += coeff_t * t[0] * contraction
    return 2*1j/len(ind_nn_i)/len(ind_nn_j)*np.imag(JJ)*Jbond**2

def jjCorrelatorBond(system,ind_i,A,B,G,H,orientation):
    """
    Compute real space <[J_i(t),J_j(0)]> correlator for a specific pair of bonds.
    Applied bond is horizontal from perturbationIndex: perturbationIndex->perturbationIndex+Ly.
    Measurement bond is specified by ind_i and orientation: h or v (right or up).
    """
    Lx = system.Lx
    Ly = system.Ly
    Jbond = system.g_i[0][ind_i,(ind_i+1)%system.Ns] if orientation=='v' else system.g_i[0][ind_i,(ind_i+Ly)%system.Ns]
    ts = system.ts
    site0 = system.site0
    indexesMap = system.indexToSite
    measurementIndex = indexesMap.index(system._xy(ind_i))
    perturbationIndex = system.perturbationIndex
    S = system.S
    magnonModes = system.p.cor_magnonModes
    #
    ts_i = ts[(site0+indexesMap[measurementIndex][0]+indexesMap[measurementIndex][1])%2]
    ts_j = ts[(site0+indexesMap[perturbationIndex][0]+indexesMap[perturbationIndex][1])%2]
    ind_r = ind_i+1 if orientation=='v' else ind_i+Ly   #-> from i
    ind_s = perturbationIndex+Ly #if system.perturbationDirection=='h' else perturbationIndex-Ly#always horizontal      -> from j
    term_list = ['XYXY','ZYZY','XYYX','ZYYZ','YXXY','YZZY','YXYX','YZYZ']
    irx,iry = system._xy(ind_r)
    ts_r = ts[(site0+irx+iry)%2]
    isx,isy = system._xy(ind_s)
    ts_s = ts[(site0+isx+isy)%2]
    ts_list = [ts_i,ts_r,ts_j,ts_s]
    JJ = np.zeros(A[0,0].shape,dtype=complex)
    for i,ops in enumerate(term_list):
        original_op = ops if i%2==0 else term_list[i-1]
        list_terms = computeCombinations(ops,[ind_i,ind_r,perturbationIndex,ind_s],'tt00',S)
        coeff_t = computeCoeffT(ops,original_op,ts_list)
        if i in [2,3,4,5]:
            coeff_t *= -1
        for t in list_terms:
            contraction = computeContraction(t[1],t[2],t[3],A,B,G,H,magnonModes)
            JJ += coeff_t * t[0] * contraction
    return 2*1j*np.imag(JJ)*Jbond**2

def generatePairings(elements):
    """
    Here we get all the possible permutation lists for the Wick contraction -> perfect matchings.
    """
    if len(elements) == 0:
        return [[]]
    pairings = []
    a = elements[0]
    for i in range(1, len(elements)):
        b = elements[i]
        rest = elements[1:i] + elements[i+1:]
        for rest_pairing in generatePairings(rest):
            pairings.append([(a, b)] + rest_pairing)
    return pairings

permutationLists = {}
for i in range(2,16,2):
    permutationLists[i] = generatePairings(list(range(i)))

def computeContraction(op_list,ind_list,time_list,A,B,G,H,magnonModes):
    """
    Here we compute the contractions using Wick decomposition of the single operator list `op_list`, with the given sites and times.
    First we compute all the 2-operator terms.
    len(op) = 2 -> 1 term
    len(op) = 4 -> 3 terms
    len(op) = 6 -> 15 terms
    len(op) = 8 -> 105 terms
    etc..
    """
    ops_dic = {'aa':B,'bb':A,'ab':H,'ba':G}
    if len(op_list)//2 in magnonModes:
        perm_list = permutationLists[len(op_list)]
        result = 0
        for i in range(len(perm_list)):
            temp = 1
            for j in range(len(perm_list[i])):
                op_ = op_list[perm_list[i][j][0]]+op_list[perm_list[i][j][1]]
                ind_ =  [ ind_list[perm_list[i][j][0]], ind_list[perm_list[i][j][1]] ]
                time_ = time_list[perm_list[i][j][0]]!=time_list[perm_list[i][j][1]]
                op = ops_dic[op_][ind_[0],ind_[1]]
                temp *= op if time_ else op[0]
            result += temp
        return result
    else:
        return 0

def computeCoeffT(op_t,op_o,ts_list):
    """
    Compute the product of t-coefficients given the original operator `op_o` and the transformed one `op_t`
    """
    ind_t_dic = {'Z':0, 'X':1, 'Y':2}
    ind_o_dic = {'X':0, 'Y':1, 'Z':2}
    coeff = 1
    for i in range(len(op_t)):
        ind_transformed = ind_t_dic[op_t[i]]
        ind_original = ind_o_dic[op_o[i]]
        coeff *= ts_list[i][ind_transformed][ind_original]
    return coeff

def computeCombinations(op_list,ind_list,time_list,S):
    """ Here we compute symbolically all terms of the HP expansion of the operator list `op`.
    a -> a, a^dag -> b.
    Need also to keep sites information
    X_j = sqrt(S/2)(a_j+b_j)
    Y_j = -i*sqrt(S/2)(a_j-b_j)
    Z_j = S-b_ja_j
    return a list of 3-tuple, with first element a coefficient (pm (i) S**n) second element an operator list ('abba..') and third element a list of sites ([ind_i,ind_j,..]) of same length as the operator string
    """
    op_dic = {'X':[np.sqrt(S/2),'a'], 'Y':'(a-b)', 'Z':'S-ba'}
    coeff_dic = {'X':np.sqrt(S/2), 'Y':1j*np.sqrt(S/2), 'Z':1}
    terms = []
    coeff = 1
    for i in range(len(op_list)):
        if op_list[i]=='X':
            terms.append([ [np.sqrt(S/2),'a',[ind_list[i]], time_list[i]] , [np.sqrt(S/2),'b',[ind_list[i]], time_list[i] ]])
        if op_list[i]=='Y':
            terms.append([ [-1j*np.sqrt(S/2),'a',[ind_list[i]], time_list[i]] , [1j*np.sqrt(S/2),'b',[ind_list[i] ], time_list[i] ]])
        if op_list[i]=='Z':
            terms.append([ [S,'',[],''] , [-1,'ba',[ind_list[i],ind_list[i]], time_list[i]+time_list[i] ]])
    for i in range(len(op_list)-1): #n-1 multiplications
        new_terms = []
        mult = []
        for j in range(len(terms[0])):
            for l in range(len(terms[1])):
                mult.append( [terms[0][j][0]*terms[1][l][0], terms[0][j][1]+terms[1][l][1], terms[0][j][2]+terms[1][l][2], terms[0][j][3]+terms[1][l][3] ]  )
        new_terms.append(mult)
        #remaining part
        for j in range(2,len(terms)):
            new_terms.append(terms[j])
        terms = list(new_terms)
    return terms[0]

def get_nn(ind,Lx,Ly):
    """ Compute indices of nearest neighbors of site ind.
    """
    result= []
    if ind+Ly<=(Lx*Ly-1):        #right neighbor
        result.append(ind+Ly)
    if ind-Ly>=0:
        result.append(ind-Ly)   #left neighbor
    if (ind+1)//Ly==ind//Ly:    #up neighbor
        result.append(ind+1)
    if (ind-1)//Ly==ind//Ly:    #bottom neighbor
        result.append(ind-1)
    return result

def get_nnn(ind,Lx,Ly):
    """ Compute indices of next-nearest neighbors of site ind.
    """
    result= []
    if ind//Ly!=Lx-1 and ind%Ly!=Ly-1:        #right-up neighbor
        result.append(ind+Ly+1)
    if ind//Ly!=Lx-1 and ind%Ly!=0:        #right-down neighbor
        result.append(ind+Ly-1)
    if ind//Ly!=0 and ind%Ly!=Ly-1:        #left-up neighbor
        result.append(ind-Ly+1)
    if ind//Ly!=0 and ind%Ly!=0:        #left-down neighbor
        result.append(ind-Ly-1)
    return result

dicCorrelators = {
    'zz':zzCorrelator,
    'ze':zeCorrelator,
    'ez':ezCorrelator,
    'ee':eeCorrelator,
    'xx':xxCorrelator,
    'jj':jjCorrelator,
}
