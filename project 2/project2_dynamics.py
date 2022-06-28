import numpy as np
from math import sin
from math import cos
from project2_CL import ConcurrentLearning

np.random.seed(0)

class Dynamics():
    def __init__(self, alpha=0.25 * np.ones(2, dtype=np.float32), beta=0.1 * np.ones(2, dtype=np.float32),
                 gamma=0.01 * np.ones(5, dtype=np.float32), lambdaCL=0.1, YYminDiff=0.1, kCL=0.9,
                 tauN=1 * np.ones(2, dtype=np.float32),
                 phiN=0.04 * np.ones(2, dtype=np.float32),
                 phiDN=0.05 * np.ones(2, dtype=np.float32),
                 phiDDN=0.06 * np.ones(2, dtype=np.float32),
                 addNoise=False):
        # noise parameters
        self.addNoise = addNoise
        self.tauN = tauN
        self.phiN = phiN
        self.phiDN = phiDN
        self.phiDDN = phiDDN
        # designing gains
        self.alpha = np.diag(alpha)
        self.beta = np.diag(beta)
        self.Gamma = np.diag(gamma)
        self.kCL = kCL
        # rigid body parameters and bounds
        self.m = np.array([2.0, 2.0], dtype=np.float32)
        self.l = np.array([0.5, 0.5], dtype=np.float32)
        self.mBnds = np.array([1.0, 3.0], dtype=np.float32)
        self.lBnds = np.array([0.25, 0.75], dtype=np.float32)
        self.g = 9.8
        # unknown parameters
        self.theta = self.getTheta(self.m, self.l)
        self.thetaH = self.getTheta(self.mBnds[0] * np.ones(2, dtype=np.float32), self.lBnds[0] * np.ones(2, dtype=np.float32))
        self.thetaHm = self.getTheta(self.mBnds[0] * np.ones(2, dtype=np.float32), self.lBnds[0] * np.ones(2, dtype=np.float32))
        # CL
        self.ConcurrentLearning = ConcurrentLearning(lambdaCL, YYminDiff)
        self.ConcurrentLearningm = ConcurrentLearning(lambdaCL, YYminDiff)
        self.tau = np.zeros(2, np.float32)
        # desired trajectory
        self.phidMag = np.array([np.pi / 8, np.pi / 4], dtype=np.float32)
        self.fphid = 0.2
        self.aphid = np.pi / 2
        self.bphid = np.array([np.pi / 2, np.pi / 4], dtype=np.float32)
        # initial states
        self.phi, _, _ = self.getDesiredState(0.0)
        self.phim = self.phi.copy()
        self.phiD = np.zeros(2, dtype=np.float32)
        self.phiDm = np.zeros(2, dtype=np.float32)
        self.phiDD = np.zeros(2, dtype=np.float32)
        self.phiDDm = np.zeros(2, dtype=np.float32)

    def getTheta(self, m, l):
        theta = np.array([(m[0] + m[1]) * l[0] ** 2 + m[1] * l[1] ** 2,
                          m[1] * l[0] * l[1],
                          m[1] * l[1] ** 2,
                          (m[0] + m[1]) * l[0],
                          m[1] * l[1]], dtype=np.float32)
        return theta

    def getDesiredState(self, t):
        phid = np.array([self.phidMag[0] * sin(2 * np.pi * self.fphid * t - self.aphid) - self.bphid[0],
                         self.phidMag[1] * sin(2 * np.pi * self.fphid * t - self.aphid) + self.bphid[1]],
                        dtype=np.float32)

        phiDd = np.array([2 * np.pi * self.fphid * self.phidMag[0] * cos(2 * np.pi * self.fphid * t - self.aphid),
                          2 * np.pi * self.fphid * self.phidMag[1] * cos(2 * np.pi * self.fphid * t - self.aphid)],
                         dtype=np.float32)

        phiDDd = np.array(
            [-((2 * np.pi * self.fphid) ** 2) * self.phidMag[0] * sin(2 * np.pi * self.fphid * t - self.aphid),
             -((2 * np.pi * self.fphid) ** 2) * self.phidMag[1] * sin(2 * np.pi * self.fphid * t - self.aphid)],
            dtype=np.float32)

        return phid, phiDd, phiDDd

    def getM(self, m, l, phi, phim):
        m1 = m[0]
        m2 = m[1]
        l1 = l[0]
        l2 = l[1]
        c2 = cos(phi[1])
        M = np.array([[m1 * l1 ** 2 + m2 * (l1 ** 2 + 2 * l1 * l2 * c2 + l2 ** 2), m2 * (l1 * l2 * c2 + l2 ** 2)],
                      [m2 * (l1 * l2 * c2 + l2 ** 2), m2 * l2 ** 2]], dtype=np.float32)
        c2m = cos(phim[1])
        Mm = np.array([[m1 * l1 ** 2 + m2 * (l1 ** 2 + 2 * l1 * l2 * c2m + l2 ** 2), m2 * (l1 * l2 * c2m + l2 ** 2)],
                      [m2 * (l1 * l2 * c2m + l2 ** 2), m2 * l2 ** 2]], dtype=np.float32)
        return M, Mm

    def getC(self, m, l, phi, phiD, phim, phiDm):

        m1 = m[0]
        m2 = m[1]
        l1 = l[0]
        l2 = l[1]

        s2 = sin(phi[1])
        phi1D = phiD[0]
        phi2D = phiD[1]
        C = np.array([-2 * m2 * l1 * l2 * s2 * phi1D * phi2D - m2 * l1 * l2 * s2 * phi2D ** 2,
                      m2 * l1 * l2 * s2 * phi1D ** 2], dtype=np.float32)
        s2m = sin(phim[1])
        phi1Dm = phiDm[0]
        phi2Dm = phiDm[1]
        Cm = np.array([-2 * m2 * l1 * l2 * s2m * phi1Dm * phi2Dm - m2 * l1 * l2 * s2m * phi2Dm ** 2,
                      m2 * l1 * l2 * s2m * phi1Dm ** 2], dtype=np.float32)
        return C, Cm

    def getG(self, m, l, phi, phim):
        m1 = m[0]
        m2 = m[1]
        l1 = l[0]
        l2 = l[1]

        c1 = cos(phi[0])
        c12 = cos(phi[0] + phi[1])
        G = np.array([(m1 + m2) * self.g * l1 * c1 + m2 * self.g * l2 * c12,
                      m2 * self.g * l2 * c12], dtype=np.float32)

        c1m = cos(phim[0])
        c12m = cos(phim[0] + phim[1])
        Gm = np.array([(m1 + m2) * self.g * l1 * c1m + m2 * self.g * l2 * c12m,
                      m2 * self.g * l2 * c12m], dtype=np.float32)
        return G, Gm

    def getYM(self, vphi, phi, vphim, phim):

        vphi1 = vphi[0]
        vphi2 = vphi[1]
        c2 = cos(phi[1])
        YM = np.array([[vphi1, 2 * c2 * vphi1 + c2 * vphi2, vphi2, 0.0, 0.0],
                       [0.0, c2 * vphi1, vphi1 + vphi2, 0.0, 0.0]], dtype=np.float32)

        vphim1 = vphim[0]
        vphim2 = vphim[1]
        c2m = cos(phim[1])
        YMm = np.array([[vphim1, 2 * c2m * vphim1 + c2m * vphim2, vphim2, 0.0, 0.0],
                       [0.0, c2m * vphim1, vphim1 + vphim2, 0.0, 0.0]], dtype=np.float32)
        return YM, YMm

    def getYC(self, phi, phiD, phim, phiDm):

        s2 = sin(phi[1])
        phi1D = phiD[0]
        phi2D = phiD[1]
        YC = np.array([[0.0, -2 * s2 * phi1D * phi2D - s2 * phi2D ** 2, 0.0, 0.0, 0.0],
                       [0.0, s2 * phi1D ** 2, 0.0, 0.0, 0.0]], dtype=np.float32)

        s2m = sin(phim[1])
        phi1Dm = phiDm[0]
        phi2Dm = phiDm[1]
        YCm = np.array([[0.0, -2 * s2m * phi1Dm * phi2Dm - s2m * phi2Dm ** 2, 0.0, 0.0, 0.0],
                       [0.0, s2m * phi1Dm ** 2, 0.0, 0.0, 0.0]], dtype=np.float32)
        return YC, YCm

    def getYG(self, phi, phim):

        c1 = cos(phi[0])
        c12 = cos(phi[0] + phi[1])
        YG = np.array([[0.0, 0.0, 0.0, self.g * c1, self.g * c12],
                       [0.0, 0.0, 0.0, 0.0, self.g * c12]], dtype=np.float32)

        c1m = cos(phim[0])
        c12m = cos(phim[0] + phim[1])
        YGm = np.array([[0.0, 0.0, 0.0, self.g * c1m, self.g * c12m],
                       [0.0, 0.0, 0.0, 0.0, self.g * c12m]], dtype=np.float32)
        return YG, YGm

    def getYMD(self, phi, phim, phiD, phiDm, r, rm):
        s2 = sin(phi[1])
        phi2D = phiD[1]
        r1 = r[0]
        r2 = r[1]
        YMD = np.array([[0.0, -2 * s2 * phi2D * r1 - s2 * phi2D * r2, 0.0, 0.0, 0.0],
                        [0.0, -s2 * phi2D * r1, 0.0, 0.0, 0.0]], dtype=np.float32)

        s2m = sin(phim[1])
        phi2Dm = phiDm[1]
        r1m = rm[0]
        r2m = rm[1]
        YMDm = np.array([[0.0, -2 * s2m * phi2Dm * r1m - s2m * phi2Dm * r2m, 0.0, 0.0, 0.0],
                        [0.0, -s2m * phi2Dm * r1m, 0.0, 0.0, 0.0]], dtype=np.float32)
        return YMD, YMDm

    def getState(self, t):
        phim = self.phim+self.phiN*np.random.randn()
        phiDm = self.phiDm+self.phiDN*np.random.randn()
        phiDDm = self.phiDDm+self.phiDDN*np.random.randn()
        return self.phi, phim, self.phiD, phiDm, self.phiDD, phiDDm, self.thetaH, self.thetaHm, self.theta

    def getErrorState(self, t):
        phid, phiDd, _ = self.getDesiredState(t)
        # return phid, phiDd, phiDDd

        e = phid - self.phi
        eD = phiDd - self.phiD
        r = eD + self.alpha @ e
        thetaTilde = self.theta - self.thetaH

        _, phim, _, phiDm, _, _, _, thetaHm, _ = self.getState(t)
        # return self.phi, phim, self.phiD, phiDm, self.phiDD, phiDDm, self.thetaH, self.thetaHm, self.theta
        em = phid - phim
        eDm = phiDd - phiDm
        rm = eDm + self.alpha @ em
        thetaTildem = self.theta - thetaHm

        return e, em, eD, eDm, r, rm, thetaTilde, thetaTildem

    def getCLstate(self):
        YYsumMinEig, TCL, YYsum, YtauSum = self.ConcurrentLearning.getState()
        YYsumMinEigm, TCLm, YYsumm, YtauSumm = self.ConcurrentLearningm.getState()
        return YYsumMinEig, YYsumMinEigm, TCL, TCLm, YYsum, YYsumm, YtauSum, YtauSumm

    def getTauThetaHD(self, t):
        # desired states
        _, _, phiDDd = self.getDesiredState(t)

        # error states
        e, em, eD, eDm, r, rm, _, _ = self.getErrorState(t)

        # regressor
        vphi = phiDDd + self.alpha @ eD
        vphim = phiDDd + self.alpha @ eDm

        # states
        _, phim, _, phiDm, _, _, _, _, _ = self.getState(t)
        YM, YMm = self.getYM(vphi, self.phi, vphim, phim)
        # getYM(self, vphi, phi, vphim, phim):
        # return YM, YMm

        YC, YCm = self.getYC(self.phi, self.phiD, phim, phiDm)
        # getYC(self, phi, phiD, phim, phiDm):
        # return YC, YCm

        YG, YGm = self.getYG(self.phi, phim)
        # getYG(self, phi, phim):
        # return YG, YGm

        YMD, YMDm = self.getYMD(self.phi, phim, self.phiD, phiDm, r, rm)
        # getYMD(self, phi, phim, phiD, phiDm, r, rm)

        Y = YM + YC + YG + 0.5 * YMD
        # controller
        tauff = Y @ self.thetaH
        taufb = e + self.beta @ r
        tau = tauff + taufb

        Ym = YMm + YCm + YGm + 0.5 * YMDm
        # controller with noise
        tauffm = Ym @ self.thetaHm
        taufbm = em + self.beta @ rm
        taum = tauffm + taufbm

        # update the CL stack and the update law
        YYsumMinEig, _, YYsum, YtauSum = self.ConcurrentLearning.getState()
        thetaCL = np.zeros_like(self.theta, np.float32)
        if YYsumMinEig > 0.001:
            thetaCL = np.linalg.inv(YYsum) @ YtauSum
        thetaHD = self.Gamma @ Y.T @ r + self.kCL * self.Gamma @ (YtauSum - YYsum @ self.thetaH)

        YYsumMinEigm, _, YYsumm, YtauSumm = self.ConcurrentLearningm.getState()
        thetaCLm = np.zeros_like(self.theta, np.float32)
        if YYsumMinEigm > 0.001:
            thetaCLm = np.linalg.inv(YYsumm) @ YtauSumm
        thetaHDm = self.Gamma @ Ym.T @ rm + self.kCL * self.Gamma @ (YtauSumm - YYsumm @ self.thetaHm)

        return tau, taum, thetaHD, thetaHDm, tauff, tauffm, taufb, taufbm, thetaCL, thetaCLm

    def step(self, dt, t):

        # dynamics
        M, Mm = self.getM(self.m, self.l, self.phi, self.phim)
        # getM(self, m, l, phi, phim)
        C, Cm = self.getC(self.m, self.l, self.phi, self.phiD, self.phim, self.phiDm)
        # getC(self, m, l, phi, phiD, phim, phiDm)
        G, Gm = self.getG(self.m, self.l, self.phi, self.phim)
        # getG(self, m, l, phi, phim)

        tau, taum, thetaHD, thetaHDm, _, _, _, _, _, _ = self.getTauThetaHD(t)

        # calculate the dynamics using the controller
        # Euler Method
        self.phiDD = np.linalg.inv(M) @ (-C - G + tau)
        self.phi += dt * self.phiD
        self.phiD += dt * self.phiDD
        self.thetaH += dt * thetaHD

        # Euler Method with noise
        self.phiDDm = np.linalg.inv(Mm) @ (-Cm - Gm + taum+self.tauN*np.random.randn() ) # noise
        self.phim += dt * self.phiDm
        self.phiDm += dt * self.phiDDm
        self.thetaHm += dt * thetaHDm

        # new states
        phi, phim, phiD, phiDm, phiDD, phiDDm, _, _, _ = self.getState(t)

        # CL
        YMCL, YMCLm = self.getYM(phiDD, phi, phiDDm, phim)
        # getYM(self, vphi, phi, vphim, phim):
        # return YM, YMm
        YC, YCm = self.getYC(phi, phiD, phim, phiDm)
        # getYC(self, phi, phiD, phim, phiDm):
        # return YC, YCm
        YG, YGm = self.getYG(phi, phim)
        # getYG(self, phi, phim):
        # return YG, YGm
        YCL = YMCL + YC + YG
        YCLm = YMCLm + YCm + YGm
        self.ConcurrentLearning.append(YCL, tau, t + dt)
        self.ConcurrentLearningm.append(YCLm, taum, t + dt)
