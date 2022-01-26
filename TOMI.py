from numpy import einsum, log2, average, conjugate, sqrt,trace, linspace, sin, cos, pi, eye,log, kron, array, ones, arange,zeros
import numpy as np
from numpy.random import randn
from scipy.linalg import expm,logm, qr
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt
from time import time; start=time()

''' code to calculate TOMI of a spin 1/2 spin chain

conventions (i.e. definitions of A,B,C,D partitions)
are the same as in the "chaos in quantum channels" paper

#############################################
here are the parameters that matter the most
change these to run different cases

Hamiltonian is by default tilted Ising;
can also do XXZ by using the H_XXZ function (see below)'''

N = 7 #number of sites
#if using SYK model: N is 1/2 the number of majoranas
maxTime = 20 #max time to go up to
numTimeSteps = 100 #number of steps (to produce plot)

'''
   C   D
 |||| ||||
 _________
|    U    |
|_________|
 |||| ||||
  A     B

(time arrow points up in this picture, unlike in the paper)

define the partitions
note that we must have Asize+Bsize = Csize+Dsize = N

the TOMI we calculate is:
I_3(A,C,D) = I(A,C) + I(A,D) - I(A,C union D)
where I( , ) is the mutual information
'''
Asize = 1; Bsize = N-1; Csize = N-1; Dsize = 1

############################################################################
############################################################################

#Identity matrix over n sites
def I(n):
    return eye(2**n)

#pauli matrices
sx = array([[0, 1],
            [1, 0]])

sy = array([[0 ,-1j],
            [1j, 0]])

sz = array([[1, 0],
            [0,-1]])

sxsx = kron(sx,sx)
sysy = kron(sy,sy)
szsz = kron(sz,sz)

'''
Hamiltonian
below are functions to create the Hamiltonian (as a 2^n x 2^n matrix)

for the (tilted) Ising model we have
H = - Sum [ J(sigmaZ sigmaZ) - g (sin(theta)sigmaX + cos(theta) sigmaZ) ]

or H = - Sum J(sigmaZ sigmaZ) - ( g sigmaX + h sigmaZ)

for XXZ we have
H = Sum [ sigmaX sigmaX + sigmaY sigmaY + g*sigmaZ sigma Z ]
'''
def H_IsingTheta(theta,J=1,g=1):
    H = zeros([2**N,2**N])
    #then add each term one by one
    for n in range(N-1):
        #I(N) \otimes sz \otimes sz \otimes I(N-n-2)
        H -= J* kron(I(n),kron(szsz,I(N-n-2)))

    for n in range(N):
        H -= g*kron(I(n),kron(sin(theta)*sx+cos(theta)*sz,I(N-n-1)))
    return H

def H_Ising(g,h):
    H = zeros([2**N,2**N])
    for n in range(N-1):
        H -= kron(I(n),kron(szsz,I(N-n-2)))
    for n in range(N):
        H += kron(I(n),kron(h*sz+g*sx,I(N-n-1)))
    return H

def H_XXZ(g):
    H = zeros([2**N,2**N])*(1+0j)
    for n in range(N-1):
        H += kron(I(n),kron(sxsx+sysy+g*szsz,I(N-n-2)))
    return H
#######################################################################
#rho = reduced density matrix
def entropy(rho):
    n = 2 #nth Renyi entropy
    return (1/(1-n))*log2(trace(matrix_power(rho,n)))#Renyi entropy
    #return -trace(rho@logm(rho)) #use this line instead, for von Neumann

##############################################################
##############################################################

''' main part of the code
calculate TOMI at each time step

function to calculate TOMI given U: '''
def TOMI(U):
    U *= 2**(-N/2) #normalize
    U = U.reshape(2**Csize,2**Dsize, 2**Asize,2**Bsize) #reshape into 4-leg tensor

    #calculate the reduced density matrices by partial tracing
    rhoA = einsum('ijak,ijbk',U, conjugate(U)).reshape(2**Asize,2**Asize)
    rhoB = einsum('ijka,ijkb',U, conjugate(U)).reshape(2**Bsize,2**Bsize)
    rhoC = einsum('aijk,bijk',U, conjugate(U)).reshape(2**Csize,2**Csize)
    rhoD = einsum('iajk,ibjk',U,conjugate(U)).reshape(2**Dsize,2**Dsize)


    rhoCD = einsum('ijab,ijcd',U, conjugate(U)).reshape(2**(Asize+Bsize),2**(Asize+Bsize))

    #there are different ways of calculating rhoAC and rhoAD, which may be more or less
    #efficient, depending on the partitions
    if Asize+Csize<=Bsize+Dsize:
        rhoAC = einsum('aibj,cidj',U, conjugate(U)).reshape(2**(Asize+Csize),2**(Asize+Csize))
    else:
        #this is actually rhoBD, whose entropy is equivalent to rhoAC
        rhoAC = einsum('iajb,icjd',U, conjugate(U)).reshape(2**(Bsize+Dsize),2**(Bsize+Dsize))

    if Bsize+Csize<Asize+Dsize:
        #this is actually rhoBC, whose entropy is equivalent to rhoAD
        rhoAD = einsum('aijb,cijd',U, conjugate(U)).reshape(2**(Bsize+Csize),2**(Bsize+Csize))
    else:

        rhoAD = einsum('iabj,icdj',U,conjugate(U)).reshape(2**(Asize+Dsize),2**(Asize+Dsize))

    #equivalent to rhoACD
    rhoACDbar = rhoB

    IAD = entropy(rhoA) + entropy(rhoD) - entropy(rhoAD)
    IAC = entropy(rhoA) + entropy(rhoC) - entropy(rhoAC)
    IA_CD = entropy(rhoA) + entropy(rhoCD) - entropy(rhoACDbar)

    return IAC + IAD - IA_CD




#edit this line to use a different Hamiltonian
H = H_Ising(-1.05,0.5)

timesteps = linspace(0,maxTime,numTimeSteps)
TOMIvalues = [0] #array of TOMI values

#main loop (not parallelized here, but easy to do in principle)
for t in timesteps[1:]:
    U = expm(-1j*H*t) #time evolution operator
    TOMIvalues.append(TOMI(U))


###########################################################################
'''compare with random matrix result'''
# haar random matrix
def cue(n):
    """A random matrix distribute with Haar measure of size n x n"""
    z = (randn(n,n) + 1j * randn(n,n))/sqrt(2.)
    q,r = qr(z)
    d = np.diagonal(r)
    ph = d/np.absolute(d)
    return np.multiply(q,ph)

NumSamples = 10#number of samples to average over
TOMIs_randMat = []
for i in range(NumSamples):
    TOMIs_randMat.append(TOMI(cue(2**N)))
TOMIrandMat = average(TOMIs_randMat)

plt.plot(timesteps,-TOMIrandMat*ones(len(timesteps)),label = 'random matrix')
###########################################################################

#now produce the plot
plt.plot(timesteps,-array(TOMIvalues),label = 'Tilted Ising')
plt.title('plots of -TOMI, N = '+str(N)+' spin chain')
plt.xlabel('t')
plt.ylabel('-TOMI of exp(-iHt), from 2nd Renyi entropy')
plt.legend()
#uncomment this next line to save the plot
#plt.savefig('TOMIplot',dpi=300)

print('run time = ',(time()-start)/60, " mins")
print('-TOMI(t = '+str(maxTime)+') = ',-TOMIvalues[-1].real)
print('-TOMI(random matrix) = ',-TOMIrandMat.real)
plt.show()
