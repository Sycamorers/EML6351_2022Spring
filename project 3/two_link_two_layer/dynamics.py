import numpy as np
from math import sin
from math import cos
from integral_concurrent_learning import ConcurrentLearning
from numpy.random import rand
from numpy.random import randn

np.random.seed(0)

# class for the two link dynamics
class Dynamics():
    # constructor to initialize a Dynamics object
    def __init__(self,alpha=0.2*np.identity(2),betar=0.1*np.identity(2),betadel=0.1*np.identity(2),
                 gammath=0.01,gammaw=0.01,gammav=0.01,lambdaCL=0.1,YYminDiff=0.1,kCL=0.9,
                 tauN=0.1,phiN=0.01,phiDN=0.05,phiDDN=0.05,l=5,deltaT=1.0,useCL=True,useNN=True,useYth=True,useBias=True):

        # gains
        self.ll = l
        self.L = l+1
        
        if not useBias:
            self.L -= 1
        self.alpha = alpha
        self.betar = betar
        self.betadel = betadel
        self.Gammath = gammath*np.identity(5,dtype=np.float64)
        self.Gammaw = gammaw*np.identity(self.L,dtype=np.float64)
        self.Gammav = gammav*np.identity(5,dtype=np.float64)
        self.kCL = kCL
        self.useCL = useCL
        self.useNN = useNN
        self.useYth = useYth

        # desired trajectory parameters
        self.phidMag = np.array([np.pi/8,np.pi/4],dtype=np.float64) # amplitude of oscillations in rad
        self.fphid = 0.2 # frequency in Hz
        self.aphid = np.pi/2 # phase shift in rad
        self.bphid = np.array([np.pi/2,np.pi/4],dtype=np.float64) # bias in rad

        # noise
        self.tauNM=tauN/3.0
        self.phiNM=phiN/3.0
        self.phiDNM=phiDN/3.0
        self.phiDDNM=phiDDN/3.0

        # rigid body parameters
        self.m = np.array([2.0,2.0],dtype=np.float64) # mass in kg
        self.l = np.array([0.5,0.5],dtype=np.float64) # length in m
        self.mBnds = np.array([1.0,3.0],dtype=np.float64) # mass bounds in kg
        self.lBnds = np.array([0.25,0.75],dtype=np.float64) # length bounds in m
        self.g = 9.8 # gravity in m/s^2

        # unknown structured dynamics
        self.theta = self.getTheta(self.m,self.l) # initialize theta
        self.thetaH = self.getTheta(self.mBnds[0]*np.ones(2,dtype=np.float64),self.lBnds[0]*np.ones(2,dtype=np.float64)) # initialize theta estimate to the lowerbounds

        # unknown unstructured dynamics
        #gaussian spaced throughout desired state space
        self.WH = (0.01/self.L)*randn(self.L,2)
        self.VH = (0.01/self.ll)*randn(5,self.ll)
        self.maxInner = 10.0
        
        # concurrent learning
        self.concurrentLearning = ConcurrentLearning(lambdaCL=lambdaCL,YYminDiff=YYminDiff,deltaT=deltaT,L=self.L)

        # desired trajectory parameters
        self.phidMag = np.array([np.pi/8,np.pi/4],dtype=np.float64) # amplitude of oscillations in rad
        self.fphid = 0.2 # frequency in Hz
        self.aphid = np.pi/2 # phase shift in rad
        self.bphid = np.array([np.pi/2,np.pi/4],dtype=np.float64) # bias in rad

        # initialize state
        self.phi,_,_ = self.getDesiredState(0.0) # set the initial angle to the initial desired angle
        self.phiD = np.zeros(2,dtype=np.float64)
        self.phiDD = np.zeros(2,dtype=np.float64)
        self.tau = np.zeros(2,dtype=np.float64)
        self.phiN = self.phiNM*randn()
        self.phiDN = self.phiDNM*randn()
        self.phiDDN = self.phiDDNM*randn()
        self.tauN = self.tauNM*randn()

    def getTheta(self,m,l):
        theta = np.array([(m[0]+m[1])*l[0]**2+m[1]*l[1]**2,
                          m[1]*l[0]*l[1],
                          m[1]*l[1]**2,
                          (m[0]+m[1])*l[0],
                          m[1]*l[1]],dtype=np.float64)
        return theta

    def getDesiredState(self,t):
        # desired angles
        phid = np.array([self.phidMag[0]*sin(2*np.pi*self.fphid*t-self.aphid)-self.bphid[0],
                         self.phidMag[1]*sin(2*np.pi*self.fphid*t-self.aphid)+self.bphid[1]],dtype=np.float64)

        #desired angular velocity
        phiDd = np.array([2*np.pi*self.fphid*self.phidMag[0]*cos(2*np.pi*self.fphid*t-self.aphid),
                          2*np.pi*self.fphid*self.phidMag[1]*cos(2*np.pi*self.fphid*t-self.aphid)],dtype=np.float64)

        #desired angular acceleration
        phiDDd = np.array([-((2*np.pi*self.fphid)**2)*self.phidMag[0]*sin(2*np.pi*self.fphid*t-self.aphid),
                           -((2*np.pi*self.fphid)**2)*self.phidMag[1]*sin(2*np.pi*self.fphid*t-self.aphid)],dtype=np.float64)
        
        return phid,phiDd,phiDDd

    # returns the inertia matrix
    def getM(self,m,l,phi):
        m1 = m[0]
        m2 = m[1]
        l1 = l[0]
        l2 = l[1]
        c2 = cos(phi[1])
        M = np.array([[m1*l1**2+m2*(l1**2+2*l1*l2*c2+l2**2),m2*(l1*l2*c2+l2**2)],
                      [m2*(l1*l2*c2+l2**2),m2*l2**2]],dtype=np.float64)
        return M

    # returns the centripetal coriolis matrix
    def getC(self,m,l,phi,phiD):
        m1 = m[0]
        m2 = m[1]
        l1 = l[0]
        l2 = l[1]
        s2 = sin(phi[1])
        phi1D = phiD[0]
        phi2D = phiD[1]
        C = np.array([-2*m2*l1*l2*s2*phi1D*phi2D-m2*l1*l2*s2*phi2D**2,
                      m2*l1*l2*s2*phi1D**2],dtype=np.float64)
        return C

    # returns the gravity matrix
    def getG(self,m,l,phi):
        m1 = m[0]
        m2 = m[1]
        l1 = l[0]
        l2 = l[1]
        c1 = cos(phi[0])
        c12 = cos(phi[0]+phi[1])
        G = np.array([(m1+m2)*self.g*l1*c1+m2*self.g*l2*c12,
                      m2*self.g*l2*c12],dtype=np.float64)
        return G

    # returns the inertia matrix regressor
    def getYM(self,vphi,phi):
        vphi1 = vphi[0]
        vphi2 = vphi[1]
        c2 = cos(phi[1])
        YM = np.array([[vphi1,2*c2*vphi1+c2*vphi2,vphi2,0.0,0.0],
                       [0.0,c2*vphi1,vphi1+vphi2,0.0,0.0]],dtype=np.float64)
        return YM

    # returns the centripetal coriolis matrix regressor
    def getYC(self,phi,phiD):
        s2 = sin(phi[1])
        phi1D = phiD[0]
        phi2D = phiD[1]
        YC = np.array([[0.0,-2*s2*phi1D*phi2D-s2*phi2D**2,0.0,0.0,0.0],
                       [0.0,s2*phi1D**2,0.0,0.0,0.0]],dtype=np.float64)
        return YC

    # returns the gravity matrix regressor
    def getYG(self,phi):
        c1 = cos(phi[0])
        c12 = cos(phi[0]+phi[1])
        YG = np.array([[0.0,0.0,0.0,self.g*c1,self.g*c12],
                     [0.0,0.0,0.0,0.0,self.g*c12]],dtype=np.float64)
        return YG

    # returns the inertia matrix derivative regressor
    def getYMD(self,phi,phiD,r):
        """
        Determines the inertia derivative regressor \n
        Inputs:
        -------
        \t phi:  angles \n
        \t phiD: angular velocoty \n
        \t r:    filtered tracking error \n
        
        Returns:
        -------
        \t YMD: inertia matrix derivative regressor
        """

        s2 = sin(phi[1])
        phi2D = phiD[1]
        r1 = r[0]
        r2 = r[1]
        YMD = np.array([[0.0,-2*s2*phi2D*r1-s2*phi2D*r2,0.0,0.0,0.0],
                        [0.0,-s2*phi2D*r1,0.0,0.0,0.0]],dtype=np.float64)
        return YMD

    def getsigma(self,phi,phiD,V):
        sigma = np.ones(self.L,dtype=np.float64)
        xi = np.ones(5,dtype=np.float64)
        xi[0] = phi[0]
        xi[1] = phi[1]
        xi[2] = phiD[0]
        xi[3] = phiD[1]
        Phi = V.T@xi
        for ii in range(self.ll):
            sigma[ii] = np.exp(-Phi[ii]**2)
        return sigma



    def getsigmaPrime(self,phi,phiD,V):
        """
        Determines the basis for the system \n
        Inputs:
        -------
        \t phi: position \n
        \t phiD: velocity \n
        \t V: inner weights \n
        
        Returns:
        -------
        \t sigma: basis \n
        """
        sigmaPrimeInner = np.zeros(self.ll,dtype=np.float64)
        sigmaPrime = np.zeros((self.L,self.ll),dtype=np.float64)
        xi = np.ones(5,dtype=np.float64)
        xi[0] = phi[0]
        xi[1] = phi[1]
        xi[2] = phiD[0]
        xi[3] = phiD[1]
        Phi = V.T@xi
        for ii in range(self.ll):
            sigmaPrimeInner[ii] = -2.0*Phi[ii]*np.exp(-Phi[ii]**2)
        sigmaPrime[:self.ll,:] = np.diag(sigmaPrimeInner)
        return sigmaPrime

    def getState(self,t):
        """
        Returns the state of the system and parameter estimates \n
        Inputs:
        -------
        \t t: time \n
        
        Returns:
        -------
        \t phi:    angles \n
        \t phiD:   angular velocity \n
        \t phiDD:  angular acceleration \n
        \t thetaH: structured parameter estimate \n
        \t WH: unstructured parameter estiate \n
        \t VH: unstructured parameter estiate \n
        """
        return self.phi,self.phiD,self.phiDD,self.thetaH,self.WH,self.VH

    #returns the error state
    def getErrorState(self,t):
        """
        Returns the errors \n
        Inputs:
        -------
        \t t:  time \n
        
        Returns:
        -------
        \t em:     measured tracking error \n
        \t eDm:    measured tracking error derivative \n
        \t rm:    measured filtered tracking error \n
        \t phim:     measured angle \n
        \t phiDm:    measured velocity \n
        \t phiDDm:    measured acceleration \n
        \t thetaH: structured estimate \n
        \t WH: unstructured estimate \n
        """
        
        # get the desired state
        phid,phiDd,phiDDd = self.getDesiredState(t)

        # get the tracking error
        em = phid - self.phi
        eDm = phiDd - self.phiD
        rm = eDm+self.alpha@em

        return em,eDm,rm

    def getCLstate(self):
        """
        Returns select parameters CL \n
        Inputs:
        -------
        
        Returns:
        -------
        \t YYsumMinEig: current minimum eigenvalue of sum of the Y^T*Y terms \n
        \t TCL: time of the minimum eigenvalue found \n
        \t YYsum: Y^T*Y sum \n
        \t YtauSum: Y^T*tau sum \n

        """
        YYsumMinEig,TCL,YYsum,YtauSum = self.concurrentLearning.getState()
        return YYsumMinEig,TCL,YYsum,YtauSum

    def getTau(self,t,phi,phiD,thetaH,WH,VH):
        """
        Returns tau \n
        Inputs:
        -------
        \t t: time \n
        \t phi: angles \n
        \t phiD: velocity \n
        \t thetaH: structured estimate \n
        \t WH: unstructured estimate outer \n
        \t VH: unstructured estimate inner \n
        
        Returns:
        -------
        \t tau: input \n
        \t tauff: feedforward component \n
        \t taufb: feedback component \n
        \t Y: regressor \n
        \t sigma: basis \n
        \t r: filtered error \n

        """
        #get desired state
        phid,phiDd,phiDDd = self.getDesiredState(t)

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
        sigma = self.getsigma(phi,phiD,VH)

        #calculate the controller and update law
        tauff = np.zeros(2)
        if self.useYth:
            tauff+=Y@thetaH
        if self.useNN:
            tauff+=WH.T@sigma
        taufb = e+self.betar@r+self.betadel@np.sign(r)
        tau = tauff + taufb

        return tau,tauff,taufb,Y,sigma,r

    def getTaud(self,phi,phiD):
        # return np.zeros(2)
        Taud = np.array([5*phi[0]**3+5.0*phi[0]+5*np.tanh(5*phiD[0])*phiD[0]**2+5*phiD[0]+2*self.g*cos(phi[1])+self.g*cos(phi[0]+phi[1]),
                        5*phi[1]**3+5.0*phi[1]+5*np.tanh(5*phiD[1])*phiD[1]**2+5*phiD[1]+self.g*cos(phi[0]+phi[1])], dtype=np.float64)
        return Taud

    def getfunc(self,phi,phiD,tau):
        M = self.getM(self.m,self.l,phi)
        C = self.getC(self.m,self.l,phi,phiD)
        G = self.getG(self.m,self.l,phi)
        taud = self.getTaud(phi,phiD)
        phiDD = np.linalg.inv(M)@(-C-G-taud+self.tauN+tau)
        return phiDD

    def getfuncComp(self,phi,phiD,phiDD,tau,thetaH,WH,VH):
        """
        Dynamics callback for function approx compare \n
        Inputs:
        -------
        \t x: position \n
        \t WH: estimates \n
        
        Returns:
        -------
        \t f: value of dynamics \n
        \t fH: approximate of dynamics \n
        """
        
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
        sigmam = self.getsigma(phi,phiD,VH)

        # get the function approximate
        fH = np.zeros(2)
        if self.useYth:
            fH += Y@thetaH

        if self.useNN:
            fH += WH.T@sigmam

        return f,fH

    def getf(self,t,X):
        """
        Dynamics callback \n
        Inputs:
        -------
        \t t:  time \n
        \t X:  stacked phi,phiD,thetaH \n
        
        Returns:
        -------
        \t XD: derivative approximate at time \n
        \t tau: control input at time \n
        """

        phi = X[0:2]
        phiD = X[2:4]
        thetaH = X[4:9]
        WH = np.reshape(X[9:9+2*self.L],(2,self.L)).T
        VH = np.reshape(X[9+2*self.L:],(5,self.ll))

        # get the noisy measurements for the control design
        phim = phi+self.phiN
        phiDm = phiD+self.phiDN
        xim = np.ones(5,dtype=np.float64)
        xim[0] = phim[0]
        xim[1] = phim[1]
        xim[2] = phiDm[0]
        xim[3] = phiDm[1]

        # get the input and regressors
        taum,_,_,Ym,sigmam,rm = self.getTau(t,phim,phiDm,thetaH,WH,VH)

        sigmaPm = self.getsigmaPrime(phim,phiDm,VH)

        #parameter updates
        thetaHD = self.Gammath@Ym.T@rm
        WHD = self.Gammaw@(np.outer(sigmam,rm))
        VHD = self.Gammav@np.outer(xim,rm)@(WH.T@sigmaPm)

        # get the dynamics using the unnoised position and velocity but designed input
        phiDD = self.getfunc(phi,phiD,taum)

        # # get the update law
        # _,_,YYsum,YtauSum = self.concurrentLearning.getState()
        # thetaHD = self.Gamma@Y.T@r + self.kCL*self.Gamma@(YtauSum - YYsum@thetaH)

        #calculate and return the derivative
        XD = np.zeros_like(X)
        XD[0:2] = phiD
        XD[2:4] = phiDD
        XD[4:9] = thetaHD
        XD[9:9+2*self.L] = np.reshape(WHD.T,(2*self.L))
        XD[9+2*self.L:] = np.reshape(VHD,(5*self.ll))

        return XD,taum

    #classic rk4 method
    def rk4(self,dt,t,X):
        """
        Classic rk4 method \n
        Inputs:
        -------
        \t dt:  total time step for interval \n
        \t t:  time \n
        \t X:  stacked x,WH \n
        
        Returns:
        -------
        \t XD: derivative approximate over total interval \n
        \t tau: control input approximate over total interval \n
        \t Xh: integrated value \n
        """

        k1,tau1 = self.getf(t,X)
        k2,tau2 = self.getf(t+0.5*dt,X+0.5*dt*k1)
        k3,tau3 = self.getf(t+0.5*dt,X+0.5*dt*k2)
        k4,tau4 = self.getf(t+dt,X+dt*k3)
        XD = (1.0/6.0)*(k1+2.0*k2+2.0*k3+k4)
        taum = (1.0/6.0)*(tau1+2.0*tau2+2.0*tau3+tau4)

        return XD,taum

    # take a step of the dynamics
    def step(self,dt,t):
        """
        Steps the internal state using the dynamics \n
        Inputs:
        -------
        \t dt: time step \n
        \t t:  time \n
        
        Returns:
        -------
        """
        # print("time "+str(t))
        # update the internal state
        X = np.zeros(2+2+5+2*self.L+5*self.ll,dtype=np.float64)
        X[0:2] = self.phi
        X[2:4] = self.phiD
        X[4:9] = self.thetaH
        X[9:9+2*self.L] = np.reshape(self.WH.T,(2*self.L))
        X[9+2*self.L:] = np.reshape(self.VH,(5*self.ll))

        #get the derivative and input from rk
        XD,taum = self.rk4(dt,t,X)
        phiD = XD[0:2]
        phiDD = XD[2:4]
        thetaHD = XD[4:9]
        # WHD = np.reshape(XD[9:9+2*self.L],(2,self.L)).T
        # VHD = np.reshape(XD[9+2*self.L:],(5,self.ll))

        #do the projection
        XNN = X[9:]
        XDNN = XD[9:]
        for ii in range(len(XNN)):
            if abs(XNN[ii]+dt*XDNN[ii]) > self.maxInner:
                XDNN[ii] = 0.0

        WHD = np.reshape(XDNN[0:2*self.L],(2,self.L)).T
        VHD = np.reshape(XDNN[2*self.L:],(5,self.ll))

        # update the internal state
        # X(ii+1) = X(ii) + dt*f(X)
        self.phi += dt*phiD
        self.phiD += dt*phiDD
        self.thetaH += dt*thetaHD
        self.WH += dt*WHD
        self.VH += dt*VHD

        self.phiN = self.phiNM*randn()
        self.phiDN = self.phiDNM*randn()
        self.phiDDN = self.phiDDNM*randn()
        self.tauN = self.tauNM*randn()

        # update the concurrent learning
        # get the inertia regressor for CL
        # self.concurrentLearning.append(sigmam,xDm-um,t+dt)