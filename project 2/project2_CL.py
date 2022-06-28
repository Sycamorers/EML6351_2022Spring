import numpy as np
from numpy.linalg import eigvals


class ConcurrentLearning():
    def __init__(self, lambdaCL=0.1, YYminDiff=0.1):
        self.Ybuff = []
        self.YYsum = np.zeros((5, 5), dtype=np.float32)
        self.YtauSum = np.zeros(5, dtype=np.float32)
        self.YYsumMinEig = 0.0
        self.lambdaCL = lambdaCL
        self.TCL = 0.0
        self.lambdaCLMet = False
        self.YYminDiff = YYminDiff

    def append(self, Y, tau, t):
        if not self.lambdaCLMet:
            _, YSV, _ = np.linalg.svd(Y)
            if (np.min(YSV) > self.YYminDiff) and (np.linalg.norm(tau) > self.YYminDiff):
                minDiff = 100.0
                for Yi in self.Ybuff:
                    YYdiffi = np.linalg.norm(Yi - Y)
                    if YYdiffi < minDiff:
                        minDiff = YYdiffi

                if minDiff > self.YYminDiff:
                    self.Ybuff.append(Y)
                    YY = Y.T @ Y
                    Ytau = Y.T @ tau
                    self.YYsum += YY
                    self.YtauSum += Ytau
                    self.YYsumMinEig = np.min(eigvals(self.YYsum))

                    if self.YYsumMinEig > self.lambdaCL:
                        self.TCL = t
                        self.lambdaCLMet = True
    def getState(self):
        return self.YYsumMinEig, self.TCL, self.YYsum, self.YtauSum