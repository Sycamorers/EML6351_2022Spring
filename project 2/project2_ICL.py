import numpy as np
from numpy.linalg import eigvals

from collections import namedtuple

TrainData = namedtuple('TrainData', ('Y', 'tau', 't'))


class ConcurrentLearning():

    def __init__(self, lambdaICL=0.1, YYminDiff=0.1, deltaT=0.1):

        self.deltaT = deltaT
        self.intBuff = []
        self.Ybuff = []
        self.YYsum = np.zeros((5, 5), dtype=np.float32)
        self.YtauSum = np.zeros(5, dtype=np.float32)
        self.YYsumMinEig = 0.0
        self.lambdaICL = lambdaICL
        self.TICL = 0.0
        self.lambdaICLMet = False
        self.YYminDiff = YYminDiff

    def append(self, Y, tau, t):

        if not self.lambdaICLMet:
            self.intBuff.append(TrainData(Y, tau, t))
            Yint = np.zeros((2, 5))
            tauInt = 0.0

            if len(self.intBuff) > 2:
                deltatCurr = self.intBuff[-1].t - self.intBuff[1].t
                if deltatCurr > self.deltaT:
                    self.intBuff.pop(0)

                for ii in range(len(self.intBuff) - 1):
                    dti = self.intBuff[ii + 1].t - self.intBuff[ii].t
                    YAvgi = 0.5 * (self.intBuff[ii + 1].Y + self.intBuff[ii].Y)
                    tauAvgi = 0.5 * (self.intBuff[ii + 1].tau + self.intBuff[ii].tau)
                    Yint += dti * YAvgi
                    tauInt += dti * tauAvgi
            else:
                return

            _, YSV, _ = np.linalg.svd(Y)
            if (np.min(YSV) > self.YYminDiff) and (np.linalg.norm(tauInt) > self.YYminDiff):
                minDiff = 100.0
                for Yi in self.Ybuff:
                    YYdiffi = np.linalg.norm(Yi - Yint)
                    if YYdiffi < minDiff:
                        minDiff = YYdiffi

                if minDiff > self.YYminDiff:
                    self.Ybuff.append(Yint)
                    YY = Yint.T @ Yint
                    Ytau = Yint.T @ tauInt
                    self.YYsum += YY
                    self.YtauSum += Ytau
                    self.YYsumMinEig = np.min(eigvals(self.YYsum))

                    if self.YYsumMinEig > self.lambdaICL:
                        self.TICL = t
                        self.lambdaICLMet = True

    def getState(self):
        return self.YYsumMinEig, self.TICL, self.YYsum, self.YtauSum