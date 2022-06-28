from typing import OrderedDict
import numpy as np
import torch
import torch.nn as nn
# from torch.autograd.functional import jacobian
import torch.optim as optim
from math import sin
from math import cos
from torch import randn as randn, sigmoid_
from collections import namedtuple
import copy

devicecuda = torch.device("cuda:0")
devicecpu = torch.device("cpu")
torch.set_default_dtype(torch.float)

torch.manual_seed(0)

Data = namedtuple('Data',('phi','phiD','phiDD','tau','Y'))

## takes in a module and applies the specified weight initialization
def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
        values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
    # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))
    # m.bias.data should be 0
        m.bias.data.fill_(0)

# create a class wrapper from PyTorch nn.Module to use exp
class Exp(nn.Module):
    """
    Applies exponential radial basis \n
    Returns:
    -------
    \t Exp(x) = exp(-x**2) \n
    """
    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, x):
        return torch.exp(-x**2)

class ReplayMemory():
    def __init__(self,memorySize=100,minDistance=0.01):
        self.memorySize = memorySize
        self.memory = []
        self.memoryTensor = []
        self.minDistance = minDistance

    def append(self,phi,phiD,phiDD,tau,Y):
        newData = Data(phi,phiD,phiDD,tau,Y)
        newDataTensor = torch.cat((phi,phiD,phiDD,tau),-1)

        #check if new data different enough from old data
        dataIsNew = True
        if len(self.memoryTensor) > 1:
            for dataTensor in self.memoryTensor:
                dataDiff = torch.linalg.norm(newDataTensor-dataTensor).item()
                # print(odomDiff)
                if dataDiff < self.minDistance:
                    print("DATA NOT SAVED, TOO CLOSE TO MEMORY")
                    dataIsNew = False
                    break
        if dataIsNew:
            self.memory.append(newData)
            self.memoryTensor.append(newDataTensor)
            print("DATA SAVED, MEMORY LENGTH "+str(len(self.memory)))

    def isMemoryFull(self):
        if len(self.memory) >= self.memorySize:
            return True
        else:
            return False

    def sample(self):
        return self.memory

    def clear(self):
        #only save half randomly without replacement
        indices = np.arange(self.memorySize,dtype=np.int64)
        newIdx = np.random.choice(indices,size=self.memorySize//2,replace=False)
        memoryNew = []
        memoryTensorNew = []
        for idx in newIdx:
            memoryNew.append(self.memory[idx])
            memoryTensorNew.append(self.memoryTensor[idx].clone().detach())
        self.memory.clear()
        self.memoryTensor.clear()
        self.memory = copy.deepcopy(memoryNew)
        self.memoryTensor = copy.deepcopy(memoryTensorNew)
        memoryNew.clear()
        memoryTensorNew.clear()

class UnstructuredBasis(nn.Module):
    def __init__(self,l=10,useCuda=False,lr=0.001):
        super(UnstructuredBasis, self).__init__()
        self.n = 4
        self.l = l
        self.outputSize = 4

        #create sequential network
        self.model = nn.Sequential(OrderedDict([
                                #    ('L1',nn.Linear(self.n,self.l)),
                                #    ('sigma1',Exp()),
                                #    ('L2',nn.Linear(self.l,self.l)),
                                #    ('sigma2',Exp()),
                                #    ('L3',nn.Linear(self.l,self.l)),
                                #    ('sigma3',Exp()),
                                #    ('L4',nn.Linear(self.l,self.l)),
                                #    ('sigma4',Exp()),
                                #    ('L5',nn.Linear(self.l,self.l)),
                                #    ('sigma5',Exp()),
                                #    ('L6',nn.Linear(self.l,self.l)),
                                #    ('sigma6',Exp()),
                                #    ('L7',nn.Linear(self.l,self.l)),
                                #    ('sigma7',Exp())
                                #    ('L1',nn.Linear(self.n,self.l)),
                                #    ('sigma1',nn.Tanh()),
                                #    ('L2',nn.Linear(self.l,self.l)),
                                #    ('sigma2',nn.Tanh()),
                                #    ('L3',nn.Linear(self.l,self.l)),
                                #    ('sigma3',nn.Tanh()),
                                #    ('L4',nn.Linear(self.l,self.l)),
                                #    ('sigma4',nn.Tanh()),
                                #    ('L5',nn.Linear(self.l,self.l)),
                                #    ('sigma5',nn.Tanh()),
                                #    ('L6',nn.Linear(self.l,self.l)),
                                #    ('sigma6',nn.Tanh()),
                                #    ('L7',nn.Linear(self.l,self.l)),
                                #    ('sigma7',nn.Tanh())
                                #    ('L1',nn.Linear(self.n,self.l)),
                                #    ('sigma1',nn.ReLU()),
                                #    ('L2',nn.Linear(self.l,self.l)),
                                #    ('sigma2',nn.ReLU()),
                                #    ('L3',nn.Linear(self.l,self.l)),
                                #    ('sigma3',nn.ReLU()),
                                #    ('L4',nn.Linear(self.l,self.l)),
                                #    ('sigma4',nn.ReLU()),
                                #    ('L5',nn.Linear(self.l,self.l)),
                                #    ('sigma5',nn.ReLU()),
                                #    ('L6',nn.Linear(self.l,self.l)),
                                #    ('sigma6',nn.ReLU()),
                                #    ('L7',nn.Linear(self.l,self.l)),
                                #    ('sigma7',nn.Sigmoid())
                                #    ('L1',nn.Linear(self.n,self.l)),
                                #    ('sigma1',nn.ReLU()),
                                #    ('L2',nn.Linear(self.l,self.l)),
                                #    ('sigma2',nn.ReLU()),
                                #    ('L3',nn.Linear(self.l,self.l)),
                                #    ('sigma3',nn.ReLU()),
                                #    ('L4',nn.Linear(self.l,self.l)),
                                #    ('sigma4',nn.ReLU()),
                                #    ('L5',nn.Linear(self.l,self.l)),
                                #    ('sigma5',nn.ReLU()),
                                #    ('L6',nn.Linear(self.l,self.l)),
                                #    ('sigma6',nn.ReLU()),
                                #    ('L7',nn.Linear(self.l,self.l)),
                                #    ('sigma7',nn.Tanh())
                                    ('L1',nn.Linear(self.n,self.l)),
                                   ('sigma1',nn.LeakyReLU()),
                                   ('L2',nn.Linear(self.l,self.l)),
                                   ('sigma2',nn.LeakyReLU()),
                                   ('L3',nn.Linear(self.l,self.l)),
                                   ('sigma3',nn.LeakyReLU()),
                                   ('L4',nn.Linear(self.l,self.l)),
                                   ('sigma4',nn.LeakyReLU()),
                                   ('L5',nn.Linear(self.l,self.l)),
                                   ('sigma5',nn.LeakyReLU()),
                                   ('L6',nn.Linear(self.l,self.l)),
                                   ('sigma6',nn.LeakyReLU()),
                                   ('L7',nn.Linear(self.l,self.l)),
                                   ('sigma7',nn.Tanh())
                                
                                   ]))

        self.model.apply(weights_init_normal)
        # self.loss = nn.MSELoss()
        # self.loss = nn.L1Loss()
        self.loss = nn.SmoothL1Loss()
        # self.loss = nn.BCELoss()
        # self.loss = nn.BCEWithLogitsLoss()
        # self.loss = nn.CrossEntropyLoss()
        # self.loss = nn.HingeEmbeddingLoss()
        

        #move to cuda
        if useCuda and torch.cuda.is_available():
            self.model = self.model.to(devicecuda)

        self.modelOptimizer = optim.Adam(self.model.parameters(),lr=lr)

    #implement forward pass
    def forward(self,xi):
        return self.model(xi)

# class for the two link dynamics
class Dynamics():
    # constructor to initialize a Dynamics object
    def __init__(self,alpha,betar,betadel,gammath,gammaw,tauN=0.1,phiN=0.01,phiDN=0.05,phiDDN=0.05,l=5,useNN=True,useYth=True,useBias=True,useCuda=False,memorySize=100,batchSize=10,numberEpochs=50,learnRate = 0.001):

        self.useCuda = useCuda
        # gains
        self.ll = l
        self.L = l+1

        self.memorySize = memorySize
        self.batchSize = batchSize
        self.numberEpochs = numberEpochs
        
        if not useBias:
            self.L -= 1
        self.alpha = self.usecuda(alpha)
        self.betar = self.usecuda(betar)
        self.betadel = self.usecuda(betadel)
        
        self.Gammath = self.usecuda(gammath*torch.eye(5,dtype=torch.float))
        self.Gammaw = self.usecuda(gammaw*torch.eye(self.L,dtype=torch.float))
        self.useNN = useNN
        self.useYth = useYth

        # desired trajectory parameters
        self.phidMag = self.usecuda(torch.tensor([torch.pi/8,torch.pi/4],dtype=torch.float)) # amplitude of oscillations in rad
        self.fphid = 0.2 # frequency in Hz
        self.aphid = torch.pi/2 # phase shift in rad
        self.bphid = self.usecuda(torch.tensor([torch.pi/2,torch.pi/4],dtype=torch.float)) # bias in rad

        # noise
        self.tauNM=tauN/3.0
        self.phiNM=phiN/3.0
        self.phiDNM=phiDN/3.0
        self.phiDDNM=phiDDN/3.0

        # rigid body parameters
        self.m = self.usecuda(torch.tensor([2.0,2.0],dtype=torch.float)) # mass in kg
        self.l = self.usecuda(torch.tensor([0.5,0.5],dtype=torch.float)) # length in m
        self.mBnds = self.usecuda(torch.tensor([1.0,3.0],dtype=torch.float)) # mass bounds in kg
        self.lBnds = self.usecuda(torch.tensor([0.25,0.75],dtype=torch.float)) # length bounds in m
        self.g = 9.8 # gravity in m/s^2

        # unknown structured dynamics
        self.theta = self.usecuda(self.getTheta(self.m,self.l)) # initialize theta
        self.thetaH = self.usecuda(self.getTheta(self.mBnds[0]*self.usecuda(torch.ones(2,dtype=torch.float)),self.lBnds[0]*self.usecuda(torch.ones(2,dtype=torch.float)))) # initialize theta estimate to the lowerbounds

        # unknown unstructured dynamics
        self.WH = self.usecuda((0.01/self.L)*randn(self.L,2))
        self.maxInner = 10.0
        self.model = UnstructuredBasis(l=self.ll,lr=learnRate)
        self.replayMemory = ReplayMemory(memorySize=memorySize)
        self.avgLoss = -1

        # desired trajectory parameters
        self.phidMag = self.usecuda(torch.tensor([torch.pi/8,torch.pi/4],dtype=torch.float)) # amplitude of oscillations in rad
        self.fphid = 0.2 # frequency in Hz
        self.aphid = torch.pi/2 # phase shift in rad
        self.bphid = self.usecuda(torch.tensor([torch.pi/2,torch.pi/4],dtype=torch.float)) # bias in rad

        # initialize state
        self.phi,_,_ = self.getDesiredState(0.0) # set the initial angle to the initial desired angle
        self.phi = self.usecuda(self.phi)
        self.phiD = self.usecuda(torch.zeros(2,dtype=torch.float))
        self.phiDD = self.usecuda(torch.zeros(2,dtype=torch.float))
        self.tau = self.usecuda(torch.zeros(2,dtype=torch.float))
        self.phiN = self.phiNM*self.usecuda(randn(1))
        self.phiDN = self.phiDNM*self.usecuda(randn(1))
        self.phiDDN = self.phiDDNM*self.usecuda(randn(1))
        self.tauN = self.tauNM*self.usecuda(randn(1))
    
    def usecuda(self,tensor: torch.Tensor):
        if self.useCuda and torch.cuda.is_available():
            tensor = tensor.to(devicecuda)
        else:
            tensor = tensor.to(devicecpu)
        return tensor

    def getTheta(self,m,l):
        theta = torch.tensor([(m[0]+m[1])*l[0]**2+m[1]*l[1]**2,
                          m[1]*l[0]*l[1],
                          m[1]*l[1]**2,
                          (m[0]+m[1])*l[0],
                          m[1]*l[1]],dtype=torch.float)
        return theta

    def getDesiredState(self,t):
        # desired angles
        phid = torch.tensor([self.phidMag[0]*sin(2*torch.pi*self.fphid*t-self.aphid)-self.bphid[0],
                         self.phidMag[1]*sin(2*torch.pi*self.fphid*t-self.aphid)+self.bphid[1]],dtype=torch.float)

        #desired angular velocity
        phiDd = torch.tensor([2*torch.pi*self.fphid*self.phidMag[0]*cos(2*torch.pi*self.fphid*t-self.aphid),
                          2*torch.pi*self.fphid*self.phidMag[1]*cos(2*torch.pi*self.fphid*t-self.aphid)],dtype=torch.float)

        #desired angular acceleration
        phiDDd = torch.tensor([-((2*torch.pi*self.fphid)**2)*self.phidMag[0]*sin(2*torch.pi*self.fphid*t-self.aphid),
                           -((2*torch.pi*self.fphid)**2)*self.phidMag[1]*sin(2*torch.pi*self.fphid*t-self.aphid)],dtype=torch.float)
        
        return phid,phiDd,phiDDd

    # returns the inertia matrix
    def getM(self,m,l,phi):
        m1 = m[0]
        m2 = m[1]
        l1 = l[0]
        l2 = l[1]
        c2 = cos(phi[1])
        M = torch.tensor([[m1*l1**2+m2*(l1**2+2*l1*l2*c2+l2**2),m2*(l1*l2*c2+l2**2)],
                      [m2*(l1*l2*c2+l2**2),m2*l2**2]],dtype=torch.float)
        return self.usecuda(M)

    # returns the centripetal coriolis matrix
    def getC(self,m,l,phi,phiD):
        m1 = m[0]
        m2 = m[1]
        l1 = l[0]
        l2 = l[1]
        s2 = sin(phi[1])
        phi1D = phiD[0]
        phi2D = phiD[1]
        C = torch.tensor([-2*m2*l1*l2*s2*phi1D*phi2D-m2*l1*l2*s2*phi2D**2,
                      m2*l1*l2*s2*phi1D**2],dtype=torch.float)
        return self.usecuda(C)

    # returns the gravity matrix
    def getG(self,m,l,phi):
        m1 = m[0]
        m2 = m[1]
        l1 = l[0]
        l2 = l[1]
        c1 = cos(phi[0])
        c12 = cos(phi[0]+phi[1])
        G = torch.tensor([(m1+m2)*self.g*l1*c1+m2*self.g*l2*c12,
                      m2*self.g*l2*c12],dtype=torch.float)
        return self.usecuda(G)

    # returns the inertia matrix regressor
    def getYM(self,vphi,phi):
        vphi1 = vphi[0]
        vphi2 = vphi[1]
        c2 = cos(phi[1])
        YM = torch.tensor([[vphi1,2*c2*vphi1+c2*vphi2,vphi2,0.0,0.0],
                       [0.0,c2*vphi1,vphi1+vphi2,0.0,0.0]],dtype=torch.float)
        return self.usecuda(YM)

    # returns the centripetal coriolis matrix regressor
    def getYC(self,phi,phiD):
        s2 = sin(phi[1])
        phi1D = phiD[0]
        phi2D = phiD[1]
        YC = torch.tensor([[0.0,-2*s2*phi1D*phi2D-s2*phi2D**2,0.0,0.0,0.0],
                       [0.0,s2*phi1D**2,0.0,0.0,0.0]],dtype=torch.float)
        return self.usecuda(YC)

    # returns the gravity matrix regressor
    def getYG(self,phi):
        c1 = cos(phi[0])
        c12 = cos(phi[0]+phi[1])
        YG = torch.tensor([[0.0,0.0,0.0,self.g*c1,self.g*c12],
                     [0.0,0.0,0.0,0.0,self.g*c12]],dtype=torch.float)
        return self.usecuda(YG)

    # returns the inertia matrix derivative regressor
    def getYMD(self,phi,phiD,r):
        s2 = sin(phi[1])
        phi2D = phiD[1]
        r1 = r[0]
        r2 = r[1]
        YMD = torch.tensor([[0.0,-2*s2*phi2D*r1-s2*phi2D*r2,0.0,0.0,0.0],
                        [0.0,-s2*phi2D*r1,0.0,0.0,0.0]],dtype=torch.float)
        return self.usecuda(YMD)

    def getsigma(self,phi,phiD):
        with torch.no_grad():
            xi = self.usecuda(torch.ones(4,dtype=torch.float))
            xi[0] = phi[0]
            xi[1] = phi[1]
            xi[2] = phiD[0]
            xi[3] = phiD[1]
            sigma = self.model(xi)
        return sigma

    def getState(self,t):
        return self.phi,self.phiD,self.phiDD,self.thetaH,self.WH

    #returns the error state
    def getErrorState(self,t):
        
        # get the desired state
        phid,phiDd,phiDDd = self.getDesiredState(t)

        phid = self.usecuda(phid)
        phiDd = self.usecuda(phiDd)
        phiDDd = self.usecuda(phiDDd)

        # get the tracking error
        em = phid - self.phi
        eDm = phiDd - self.phiD
        rm = eDm+self.alpha@em

        return em,eDm,rm

    def getTau(self,t,phi,phiD,thetaH,WH):

        #get desired state
        phid,phiDd,phiDDd = self.getDesiredState(t)
        phid = self.usecuda(phid)
        phiDd = self.usecuda(phiDd)
        phiDDd = self.usecuda(phiDDd)

        #calculate error
        e = phid - phi
        eD = phiDd - phiD
        r = eD + self.alpha@e
        vphi = phiDDd + self.alpha@eD

        # get the regressors
        vphi = phiDDd + self.alpha@eD
        YM = self.getYM(vphi,phi)
        YC = self.getYC(phi,phiD)
        YG = self.getYG(phi)
        YMD = self.getYMD(phi,phiD,r)
        Y = YM+YC+YG+0.5*YMD

        #get sigma
        sigma = self.getsigma(phi,phiD)

        #calculate the controller and update law
        tauff = self.usecuda(torch.zeros(2))
        if self.useYth:
            tauff+=Y@thetaH
        if self.useNN:
            tauff+=WH.T@sigma
        taufb = e+self.betar@r+self.betadel@torch.sign(r)
        tau = tauff + taufb

        return tau,tauff,taufb,Y,sigma,r

    def getTaud(self,phi,phiD):
        # return np.zeros(2)
        taudFriction = self.usecuda(torch.tensor([5.0*torch.tanh(5.0*phiD[0])*phiD[0]**2+5.0*phiD[0],5.0*torch.tanh(5.0*phiD[1])*phiD[1]**2+5.0*phiD[1]]))
        taudSpring = self.usecuda(torch.tensor([5.0*phi[0]**3+5.0*phi[0],5.0*phi[1]**3+5.0*phi[1]]))
        taudForce = self.usecuda(torch.tensor([2.0*self.g*cos(phi[0])+self.g*cos(phi[0]+phi[1]),self.g*cos(phi[0]+phi[1])]))
        return taudFriction+taudSpring+taudForce

    def getfunc(self,phi,phiD,tau):
        M = self.getM(self.m,self.l,phi)
        C = self.getC(self.m,self.l,phi,phiD)
        G = self.getG(self.m,self.l,phi)
        taud = self.getTaud(phi,phiD)
        phiDD = torch.linalg.inv(M)@(-C-G-taud+self.tauN+tau)
        return phiDD

    def getfuncComp(self,phi,phiD,phiDD,tau,thetaH,WH):
        #calculate the actual
        M = self.getM(self.m,self.l,phi)
        C = self.getC(self.m,self.l,phi,phiD)
        G = self.getG(self.m,self.l,phi)
        taud = self.getTaud(phi,phiD)
        
        f = M@phiDD+C+G+taud

        # calculate the approximate
        # get regressors
        YM = self.getYM(phiDD,phi)
        YC = self.getYC(phi,phiD)
        YG = self.getYG(phi)
        Y = YM+YC+YG

        #get sigma
        sigmam = self.getsigma(phi,phiD)

        # get the function approximate
        fH = self.usecuda(torch.zeros(2))
        if self.useYth:
            fH+=Y@thetaH
        if self.useNN:
            fH += WH.T@sigmam

        return f,fH

    def getf(self,t,X):
        phi = X[0:2]
        phiD = X[2:4]
        thetaH = X[4:9]
        WH = torch.reshape(X[9:9+2*self.L],(2,self.L)).T

        # get the noisy measurements for the control design
        phim = phi+self.phiN
        phiDm = phiD+self.phiDN

        # get the input and regressors
        taum,_,_,Ym,sigmam,rm = self.getTau(t,phim,phiDm,thetaH,WH)

        #parameter updates
        thetaHD = self.Gammath@Ym.T@rm
        WHD = self.Gammaw@(torch.outer(sigmam,rm))

        # get the dynamics using the unnoised position and velocity but designed input
        phiDD = self.getfunc(phi,phiD,taum)

        #calculate and return the derivative
        XD = torch.zeros_like(X)
        XD[0:2] = phiD
        XD[2:4] = phiDD
        XD[4:9] = thetaHD
        XD[9:9+2*self.L] = torch.reshape(WHD.T,(2*self.L,))

        return XD,taum

    #classic rk4 method
    def rk4(self,dt,t,X):
        k1,tau1 = self.getf(t,X)
        k2,tau2 = self.getf(t+0.5*dt,X+0.5*dt*k1)
        k3,tau3 = self.getf(t+0.5*dt,X+0.5*dt*k2)
        k4,tau4 = self.getf(t+dt,X+dt*k3)
        XD = (1.0/6.0)*(k1+2.0*k2+2.0*k3+k4)
        taum = (1.0/6.0)*(tau1+2.0*tau2+2.0*tau3+tau4)

        return XD,taum

    # take a step of the dynamics
    def step(self,dt,t):
        # update the internal state
        X = self.usecuda(torch.zeros(2+2+5+2*self.L+5*self.ll,dtype=torch.float))
        X[0:2] = self.phi
        X[2:4] = self.phiD
        X[4:9] = self.thetaH
        X[9:9+2*self.L] = torch.reshape(self.WH.T,(2*self.L,))

        #get the derivative and input from rk
        XD,taum = self.rk4(dt,t,X)
        phiD = XD[0:2]
        phiDD = XD[2:4]
        thetaHD = XD[4:9]
        WHD = torch.reshape(XD[9:9+2*self.L],(2,self.L)).T

        #do the projection
        XNN = X[9:]
        XDNN = XD[9:]
        for ii in range(len(XNN)):
            if abs(XNN[ii]+dt*XDNN[ii]) > self.maxInner:
                XDNN[ii] = 0.0

        WHD = torch.reshape(XDNN[0:2*self.L],(2,self.L)).T
        
        #save the data to optimize
        YM = self.getYM(self.phiDD,self.phi)
        YC = self.getYC(self.phi,self.phiD)
        YG = self.getYG(self.phi)
        Y = YM+YC+YG
        self.replayMemory.append(self.phi,self.phiD,self.phiDD,taum,Y)
        avgLoss = self.optimize()
        if avgLoss >= 0:
            self.avgLoss = avgLoss

        # print("time: "+str(t)+" avg loss: "+str(self.avgLoss))

        # update the internal state
        # X(ii+1) = X(ii) + dt*f(X)
        self.phi += dt*phiD
        self.phiD += dt*phiDD
        self.thetaH += dt*thetaHD
        self.WH += dt*WHD

        self.phiN = self.phiNM*self.usecuda(randn(1))
        self.phiDN = self.phiDNM*self.usecuda(randn(1))
        self.phiDDN = self.phiDDNM*self.usecuda(randn(1))
        self.tauN = self.tauNM*self.usecuda(randn(1))

    def getLoss(self):
        return self.avgLoss

    def optimize(self):
        if not self.replayMemory.isMemoryFull():
            # print("MEMORY NOT FILLED")
            return -1

        # print("MEMORY FILLED OPTIMIZING BASIS")
        self.model.train()

        transitions = self.replayMemory.sample()
        sampleSize = len(transitions)

        # group the transitions into a dict of batch arrays
        batch = Data(*zip(*transitions))
        # rospy.logwarn("batch "+str(batch))

        #get the batches, push to cuda
        phiBatch = torch.cat(batch.phi)
        phiDBatch = torch.cat(batch.phiD)
        tauBatch = torch.cat(batch.tau)
        YBatch = torch.cat(batch.Y)

        if torch.cuda.is_available():
            phiBatch = phiBatch.to(devicecuda)
            phiDBatch = phiDBatch.to(devicecuda)
            tauBatch = tauBatch.to(devicecuda)
            YBatch = YBatch.to(devicecuda)
            self.thetaH = self.thetaH.to(devicecuda)
            self.WH = self.WH.to(devicecuda)
            self.model.model = self.model.model.to(devicecuda)
        
        phiBatch = phiBatch.view(sampleSize,-1)
        phiDBatch = phiDBatch.view(sampleSize,-1)
        Xbatch = torch.cat((phiBatch,phiDBatch),1)
        tauBatch = tauBatch.view(sampleSize,-1)
        YBatch = YBatch.view(sampleSize,2,5)



        #train random batches over number of epochs using MSE
        losses = []
        for _ in range(self.numberEpochs):
            # generate random set of batches
            batchStart = np.arange(0,sampleSize,self.batchSize,dtype=np.int64)
            indices = np.arange(sampleSize,dtype=np.int64)
            # np.random.shuffle(indices)
            np.random.shuffle(batchStart)
            batches = [indices[ii:ii+self.batchSize] for ii in batchStart]
            # print(batches)

            for batch in batches:
                if len(batch)  == self.batchSize:
                    Xbatchii    = Xbatch[batch,:]
                    tauBatchii  = tauBatch[batch,:]
                    YBatchii    = YBatch[batch,:]

                    tauHatBatchii = YBatchii@self.thetaH+(self.WH.T@self.model(Xbatchii).T).T
                    
                    # loss = ((tauBatchii-tauHatBatchii)**2.0).mean()
                    # loss = (torch.abs(tauBatchii-tauHatBatchii)).mean()

                    # sigmoid = nn.Sigmoid()
                    # sigmoid_input = sigmoid(tauHatBatchii)
                    # loss = self.model.loss(sigmoid_input,tauBatchii)

                    loss = self.model.loss(tauHatBatchii,tauBatchii)
                    print(loss)

                    self.model.modelOptimizer.zero_grad()
                    loss.backward()
                    self.model.modelOptimizer.step()

                    losses.append(loss.item())

            batches.clear()

        lossesAvg = np.asarray(losses).mean().item()
        losses.clear()
        self.replayMemory.clear()

        if not self.useCuda:
            self.thetaH = self.thetaH.to(devicecpu)
            self.WH = self.WH.to(devicecpu)
            self.model.model = self.model.model.to(devicecpu)

        return lossesAvg