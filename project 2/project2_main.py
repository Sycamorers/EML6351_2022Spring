import numpy as np
import project2_dynamics
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

if __name__ == '__main__':

    # noise
    addNoise = True
    tauNoise = 1*np.ones(2, dtype=np.float32)
    phiNoise = 0.04*np.ones(2, dtype=np.float32)
    phiDNoise = 0.05*np.ones(2, dtype=np.float32)
    phiDDNoise = 0.06*np.ones(2, dtype=np.float32)

    dt = 0.005
    tf = 60.0
    t = np.linspace(0.0, tf, int(tf / dt), dtype=np.float32)
    alpha = 4*np.ones(2, dtype=np.float32)
    beta = 2*np.ones(2, dtype=np.float32)
    gamma = 0.5*np.ones(5, dtype=np.float32)
    lambdaCL = 1.0
    YYminDiff = 0.1
    kCL = 0.1
    dyn = project2_dynamics.Dynamics(alpha=alpha, beta=beta, gamma=gamma, lambdaCL=lambdaCL,
                                     YYminDiff=YYminDiff, kCL=kCL,
                                     tauN=tauNoise, phiN=phiNoise, phiDN=phiDDNoise, phiDDN=phiDDNoise)

    phiHist = np.zeros((2, len(t)), dtype=np.float32)
    phimHist = np.zeros((2, len(t)), dtype=np.float32)
    phidHist = np.zeros((2, len(t)), dtype=np.float32)
    phiDHist = np.zeros((2, len(t)), dtype=np.float32)
    phiDmHist = np.zeros((2, len(t)), dtype=np.float32)
    phiDdHist = np.zeros((2, len(t)), dtype=np.float32)
    phiDDHist = np.zeros((2, len(t)), dtype=np.float32)
    phiDDmHist = np.zeros((2, len(t)), dtype=np.float32)
    phiDDdHist = np.zeros((2, len(t)), dtype=np.float32)
    eHist = np.zeros((2, len(t)), dtype=np.float32)
    emHist = np.zeros((2, len(t)), dtype=np.float32)
    eNormHist = np.zeros_like(t)
    emNormHist = np.zeros_like(t)
    rHist = np.zeros((2, len(t)), dtype=np.float32)
    rmHist = np.zeros((2, len(t)), dtype=np.float32)
    rNormHist = np.zeros_like(t)
    rmNormHist = np.zeros_like(t)
    thetaHist = np.zeros((5, len(t)), dtype=np.float32)
    thetaHHist = np.zeros((5, len(t)), dtype=np.float32)
    thetaHmHist = np.zeros((5, len(t)), dtype=np.float32)
    thetaCLHist = np.zeros((5, len(t)), dtype=np.float32)
    thetaCLmHist = np.zeros((5, len(t)), dtype=np.float32)
    thetaTildeHist = np.zeros((5, len(t)), dtype=np.float32)
    thetaTildemHist = np.zeros((5, len(t)), dtype=np.float32)
    thetaTildeNormHist = np.zeros_like(t)
    thetaTildemNormHist = np.zeros_like(t)
    lambdaCLMinmHist = np.zeros_like(t)
    lambdaCLMinHist = np.zeros((2, len(t)), dtype=np.float32)
    tauHist = np.zeros((2, len(t)), dtype=np.float32)
    taumHist = np.zeros((2, len(t)), dtype=np.float32)
    tauffHist = np.zeros((2, len(t)), dtype=np.float32)
    tauffmHist = np.zeros((2, len(t)), dtype=np.float32)
    taufbHist = np.zeros((2, len(t)), dtype=np.float32)
    taufbmHist = np.zeros((2, len(t)), dtype=np.float32)

    tauHistnorm = np.zeros_like(t)
    taumHistnorm = np.zeros_like(t)
    tauffHistnorm = np.zeros_like(t)
    tauffmHistnorm = np.zeros_like(t)
    taufbHistnorm = np.zeros_like(t)
    taufbmHistnorm = np.zeros_like(t)


    TCL = 0
    TCLindex = 0
    TCLfound = False
    TCLm = 0
    TCLmindex = 0
    TCLmfound = False

    for jj in range(0, len(t)):
        phij, phimj, phiDj, phiDmj, phiDDj, phiDDmj, thetaHj, thetaHmj, thetaj = dyn.getState(t[jj])

        phidj, phiDdj, phiDDdj = dyn.getDesiredState(t[jj])

        ej, emj, _, _, rj, rmj, thetaTildej, thetaTildemj = dyn.getErrorState(t[jj])

        tauj, taumj, _, _, tauffj, tauffmj, taufbj, taufbmj, thetaCLj, thetaCLmj = dyn.getTauThetaHD(t[jj])

        lamdaCLMinj, lamdaCLMinmj, TCLj, TCLmj, _, _, _, _ = dyn.getCLstate()

        if not TCLfound:
            if TCLj > 0:
                TCL = TCLj
                TCLindex = jj
                TCLfound = True

        if not TCLmfound:
            if TCLmj > 0:
                TCLm = TCLmj
                TCLmindex = jj
                TCLmfound = True

        phiHist[:, jj] = phij
        phidHist[:, jj] = phidj
        phiDHist[:, jj] = phiDj
        phiDdHist[:, jj] = phiDdj
        phiDDHist[:, jj] = phiDDj
        phiDDdHist[:, jj] = phiDDdj
        # error
        eHist[:, jj] = ej
        eNormHist[jj] = np.linalg.norm(ej)
        # tracking error
        rHist[:, jj] = rj
        rNormHist[jj] = np.linalg.norm(rj)
        # estimation
        thetaHist[:, jj] = thetaj
        thetaHHist[:, jj] = thetaHj
        thetaCLHist[:, jj] = thetaCLj
        thetaTildeHist[:, jj] = thetaTildej
        thetaTildeNormHist[jj] = np.linalg.norm(thetaTildej)
        lambdaCLMinHist[:, jj] = [lamdaCLMinj, lamdaCLMinmj]
        # lambdaCLMinHist[jj] = lamdaCLMinj
        # lambdaCLMinmHist[jj] = lamdaCLMinmj
        tauHist[:, jj] = tauj
        tauffHist[:, jj] = tauffj
        taufbHist[:, jj] = taufbj
        tauHistnorm[jj] = np.linalg.norm(tauj)
        tauffHistnorm[jj] = np.linalg.norm(tauffj)
        taufbHistnorm[jj] = np.linalg.norm(taufbj)

        dyn.step(dt, t[jj])

        phimHist[:, jj] = phimj
        phiDmHist[:, jj] = phiDmj
        phiDDmHist[:, jj] = phiDDmj
        emHist[:, jj] = emj
        emNormHist[jj] = np.linalg.norm(emj)
        rmHist[:, jj] = rmj
        rmNormHist[jj] = np.linalg.norm(rmj)
        thetaHmHist[:, jj] = thetaHmj
        thetaCLmHist[:, jj] = thetaCLmj
        thetaTildemHist[:, jj] = thetaTildemj
        thetaTildemNormHist[jj] = np.linalg.norm(thetaTildemj)
        taumHist[:, jj] = taumj
        tauffmHist[:, jj] = tauffmj
        taufbmHist[:, jj] = taufbmj
        taumHistnorm[jj] = np.linalg.norm(taumj)
        tauffmHistnorm[jj] = np.linalg.norm(tauffmj)
        taufbmHistnorm[jj] = np.linalg.norm(taufbmj)

    # angles
    fig1, angles = plt.subplots(2)
    angles[0].set_title('angles')
    angles[0].plot(t, phidHist[0, :], color='darkviolet')
    angles[0].plot(t, phiHist[0, :], color='limegreen')
    angles[0].plot(t, phidHist[1, :], color='darkviolet')
    angles[0].plot(t, phiHist[1, :], color='limegreen')
    angles[0].legend(["$\phi_{1d}$", "$\phi_1$", "$\phi_{1m}$", "$\phi_{2d}$", "$\phi_2$", "$\phi_{2m}$"], loc='upper right')
    # angles with noise
    angles[1].set_title('angles with noise')
    angles[1].plot(t, phidHist[0, :], color='darkviolet')
    angles[1].plot(t, phidHist[1, :], color='darkviolet')
    angles[1].plot(t, phimHist[0, :], color='limegreen')
    angles[1].plot(t, phimHist[1, :], color='limegreen')
    angles[1].legend(["$\phi_{1d}$", "$\phi_{1m}$", "$\phi_{2d}$", "$\phi_{2m}$"], loc='upper right')

    # error
    fig2, errors = plt.subplots(2)
    errors[0].set_title('errors')
    errors[0].plot(t, eHist[0, :], color='darkviolet')
    errors[0].plot(t, eHist[1, :], color='limegreen')
    errors[0].legend(["$e_1$", "$e_2$"], loc='upper right')
    errors[1].set_title('errors with noise')
    errors[1].plot(t, emHist[1, :], color='darkviolet')
    errors[1].plot(t, emHist[0, :], color='limegreen')
    errors[1].legend(["$e_{1m}$", "$e_{2m}$"], loc='upper right')

    # # error norm
    fig3, errornorms = plt.subplots(2)
    errornorms[0].set_title('error norms')
    errornorms[0].plot(t, eNormHist, color='limegreen')
    errornorms[0].legend(["$e$"], loc='upper right')
    errornorms[1].set_title('error norms with noise')
    errornorms[1].plot(t, emNormHist, color='limegreen')
    errornorms[1].legend(["$e_{m}$"], loc='upper right')

    # angular velocity
    fig4, av = plt.subplots(2)
    av[0].set_title('angular velocity')
    av[0].plot(t, phiDdHist[0, :], color='darkviolet')
    av[0].plot(t, phiDHist[0, :], color='limegreen')
    av[0].plot(t, phiDdHist[1, :], color='darkviolet')
    av[0].plot(t, phiDHist[1, :], color='limegreen')
    av[0].legend(["$\dot{\phi}_{1d}$", "$\dot{\phi}_1$",  "$\dot{\phi}_{2d}$", "$\dot{\phi}_2$"], loc='upper right')
    av[1].set_title('angular velocity with noise')
    av[1].plot(t, phiDdHist[0, :], color='darkviolet')
    av[1].plot(t, phiDHist[1, :], color='limegreen')
    av[1].plot(t, phiDmHist[0, :], color='darkviolet')
    av[1].plot(t, phiDmHist[1, :], color='limegreen')
    av[1].legend(["$\dot{\phi}_{1d}$", "$\dot{\phi}_{1m}$", "$\dot{\phi}_{2d}$", "$\dot{\phi}_{2m}$"], loc='upper right')

    # filtered error
    fig5, fe = plt.subplots(2)
    fe[0].set_title('filtered error')
    fe[0].plot(t, rHist[0, :], color='darkviolet')
    fe[0].plot(t, rHist[1, :], color='limegreen')
    fe[0].legend(["$r_1$", "$r_2$"], loc='upper right')
    fe[1].set_title('filtered error with noise')
    fe[1].plot(t, rmHist[0, :], color='darkviolet')
    fe[1].plot(t, rmHist[1, :], color='limegreen')
    fe[1].legend(["$r_{1m}$", "$r_{2m}$"], loc='upper right')

    # inputs
    fig6, tau = plt.subplots(2)
    tau[0].set_title('input without noise')
    tau[0].plot(tauHist[0, :], color='darkviolet')
    tau[0].plot(tauHist[1, :], color='darkviolet')
    tau[0].plot(tauffHist[0, :], color='limegreen')
    tau[0].plot(tauffHist[1, :], color='limegreen')
    tau[0].plot(taufbHist[0, :], color='hotpink')
    tau[0].plot(taufbHist[1, :], color='hotpink')
    tau[0].legend(['$\\tau_{1}$', "$\\tau_{2}$", "$\\tau_{ff1}$", "$\\tau_{ff2}$", "$\\tau_{fb1}$", "$\\tau_{fb2}$"], loc='upper right')
    tau[1].set_title('input with noise')
    tau[1].plot(taumHist[0, :], color='darkviolet')
    tau[1].plot(tauffmHist[0, :], color='limegreen')
    tau[1].plot(taufbmHist[0, :], color='hotpink')
    tau[1].plot(taumHist[1, :], color='darkviolet')
    tau[1].plot(tauffmHist[1, :], color='limegreen')
    tau[1].plot(taufbmHist[1, :], color='hotpink')
    tau[1].legend(['$\\tau_{1m}$', "$\\tau_{2m}$", "$\\tau_{ff1m}$", "$\\tau_{ff2m}$", "$\\tau_{fb1m}$", "$\\tau_{fb2m}$"], loc='upper right')

    # parameter estimates without noise
    fig7, pe = plt.subplots(2)
    pe[0].set_title('parameter estimates without noise')
    pe[0].plot(t, thetaHist[0, :], color='limegreen', linewidth=2, linestyle='--')
    pe[0].plot(t, thetaHist[1, :], color='lightsteelblue', linewidth=2, linestyle='--')
    pe[0].plot(t, thetaHist[2, :], color='darkviolet', linewidth=2, linestyle='--')
    pe[0].plot(t, thetaHist[3, :], color='salmon', linewidth=2, linestyle='--')
    pe[0].plot(t, thetaHist[4, :], color='deepskyblue', linewidth=2, linestyle='--')
    pe[0].plot(t, thetaHHist[0, :], color='limegreen', linewidth=2, linestyle='-')
    pe[0].plot(t, thetaHHist[1, :], color='lightsteelblue', linewidth=2, linestyle='-')
    pe[0].plot(t, thetaHHist[2, :], color='darkviolet', linewidth=2, linestyle='-')
    pe[0].plot(t, thetaHHist[3, :], color='salmon', linewidth=2, linestyle='-')
    pe[0].plot(t, thetaHHist[4, :], color='deepskyblue', linewidth=2, linestyle='-')
    pe[0].plot(t, thetaCLHist[0, :], color='limegreen', linewidth=2, linestyle='-.')
    pe[0].plot(t, thetaCLHist[1, :], color='lightsteelblue', linewidth=2, linestyle='-.')
    pe[0].plot(t, thetaCLHist[2, :], color='darkviolet', linewidth=2, linestyle='-.')
    pe[0].plot(t, thetaCLHist[3, :], color='salmon', linewidth=2, linestyle='-.')
    pe[0].plot(t, thetaCLHist[4, :], color='deepskyblue', linewidth=2, linestyle='-.')
    pe[0].legend(["$\\theta_1$", "$\\theta_2$", "$\\theta_3$", "$\\theta_4$", "$\\theta_5$", "$\hat{\\theta}_1$", "$\hat{\\theta}_2$", "$\hat{\\theta}_3$", "$\hat{\\theta}_4$", "$\hat{\\theta}_5$", "$\hat{\\theta}_{CL1}$", "$\hat{\\theta}_{CL2}$", "$\hat{\\theta}_{CL3}$", "$\hat{\\theta}_{CL4}$", "$\hat{\\theta}_{CL5}$"], loc='lower right',  ncol=3)
    pe[1].set_title('parameter estimates with noise')
    pe[1].plot(t, thetaHist[0, :], color='limegreen', linewidth=2, linestyle='--')
    pe[1].plot(t, thetaHist[1, :], color='lightsteelblue', linewidth=2, linestyle='--')
    pe[1].plot(t, thetaHist[2, :], color='darkviolet', linewidth=2, linestyle='--')
    pe[1].plot(t, thetaHist[3, :], color='salmon', linewidth=2, linestyle='--')
    pe[1].plot(t, thetaHist[4, :], color='deepskyblue', linewidth=2, linestyle='--')
    pe[1].plot(t, thetaHmHist[0, :], color='limegreen', linewidth=2, linestyle='-')
    pe[1].plot(t, thetaHmHist[1, :], color='lightsteelblue', linewidth=2, linestyle='-')
    pe[1].plot(t, thetaHmHist[2, :], color='darkviolet', linewidth=2, linestyle='-')
    pe[1].plot(t, thetaHmHist[3, :], color='salmon', linewidth=2, linestyle='-')
    pe[1].plot(t, thetaHmHist[4, :], color='deepskyblue', linewidth=2, linestyle='-')
    pe[1].plot(t, thetaCLmHist[0, :], color='limegreen', linewidth=2, linestyle='-.')
    pe[1].plot(t, thetaCLmHist[1, :], color='lightsteelblue', linewidth=2, linestyle='-.')
    pe[1].plot(t, thetaCLmHist[2, :], color='darkviolet', linewidth=2, linestyle='-.')
    pe[1].plot(t, thetaCLmHist[3, :], color='salmon', linewidth=2, linestyle='-.')
    pe[1].plot(t, thetaCLmHist[4, :], color='deepskyblue', linewidth=2, linestyle='-.')
    pe[1].legend(["$\\theta_1$", "$\\theta_2$", "$\\theta_3$", "$\\theta_4$", "$\\theta_5$", "$\hat{\\theta}_1$", "$\hat{\\theta}_2$", "$\hat{\\theta}_3$", "$\hat{\\theta}_4$", "$\hat{\\theta}_5$", "$\hat{\\theta}_{CL1}$", "$\hat{\\theta}_{CL2}$", "$\hat{\\theta}_{CL3}$", "$\hat{\\theta}_{CL4}$", "$\hat{\\theta}_{CL5}$"], loc='lower right',  ncol=3)

    # parameter estiamtes norm
    fig8, pen = plt.subplots(2)
    pen[0].set_title('parameter estimates norm without noise')
    pen[0].plot(t, thetaTildeNormHist, color='limegreen')
    pen[0].legend(["$\\tilde{\\theta}$"], loc='upper right')
    pen[1].set_title('parameter estimates norm with noise')
    pen[1].plot(t, thetaTildemNormHist, color='limegreen')
    pen[1].legend(["$\\tilde{\\theta}_m$"], loc='upper right')

    # minimum eigenvalue
    fig9, mg = plt.subplots(2)
    mg[0].set_title('minimum eigenvalue without noise')
    mg[0].plot(t, lambdaCLMinHist[0, :], color='limegreen')
    mg[0].plot([TCL, TCL], [0.0, lambdaCLMinHist[0, TCLindex]], color='black', linewidth=1, linestyle='-')
    mg[1].set_title('minimum eigenvalue without noise')
    mg[1].plot(t, lambdaCLMinHist[1, :], color='limegreen')
    mg[1].plot([TCLm, TCLm], [0.0, lambdaCLMinHist[1, TCLmindex]], color='black', linewidth=1, linestyle='--')

    # inputs
    fig10, taunorm = plt.subplots(2)
    taunorm[0].set_title('input norm without noise')
    taunorm[0].plot(t, tauHistnorm, color='darkviolet')
    taunorm[0].plot(t, tauffHistnorm, color='limegreen')
    taunorm[0].plot(t, taufbHistnorm, color='hotpink')
    taunorm[0].legend(['$\\tau$', "$\\tau_{ff}$", "$\\tau_{fb}$"], loc='upper right')
    taunorm[1].set_title('input norm with noise')
    taunorm[1].plot(t, taumHistnorm, color='darkviolet')
    taunorm[1].plot(t, tauffmHistnorm, color='limegreen')
    taunorm[1].plot(t, taufbmHistnorm, color='hotpink')
    taunorm[1].legend(['$\\tau_{m}$', "$\\tau_{ffm}$", "$\\tau_{fbm}$"], loc='upper right')
    plt.show()
