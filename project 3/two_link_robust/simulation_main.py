import numpy as np
import dynamics
import os
import datetime
import matplotlib.pyplot as plot


if __name__ == '__main__':
    tauNoise = 1.0
    phiNoise = 0.04
    phiDNoise = 0.05
    phiDDNoise = 0.06
    useYth = True
    dt = 0.01 # time step
    tf = 60.0 # final time
    t = np.linspace(0.0,tf,int(tf/dt),dtype=np.float64) # times
    alpha = 3.0*np.identity(2)
    betar = 1.5*np.identity(2)
    betaeps = 0.001*np.identity(2)
    betataud= 2*np.identity(2)
    gammath = 0.75
    gammaw = 0.75
    lambdaCL = 0.0001
    YYminDiff = 0.1
    deltaT = 1.5
    kCL = 0.3
    L = 5
    Lmod = 4*L+1
    dyn = dynamics.Dynamics(alpha=alpha,betar=betar,betaeps=betaeps, betataud=betataud, gammath=gammath,gammaw=gammaw,lambdaCL=lambdaCL,YYminDiff=YYminDiff,kCL=kCL,tauN=tauNoise,phiN=phiNoise,phiDN=phiDNoise,phiDDN=phiDDNoise,L=L,deltaT=deltaT,useYth=useYth)
    phiHist = np.zeros((2,len(t)),dtype=np.float64)
    phidHist = np.zeros((2,len(t)),dtype=np.float64)
    eHist = np.zeros((2,len(t)),dtype=np.float64)
    eNormHist = np.zeros(len(t),dtype=np.float64)
    rHist = np.zeros((2,len(t)),dtype=np.float64)
    rNormHist = np.zeros(len(t),dtype=np.float64)
    thetaHHist = np.zeros((5,len(t)),dtype=np.float64)
    WHHist = np.zeros((2*Lmod,len(t)),dtype=np.float64)
    lambdaCLMinHist = np.zeros(len(t),dtype=np.float64)
    tauHist = np.zeros((2,len(t)),dtype=np.float64)
    tauffHist = np.zeros((2,len(t)),dtype=np.float64)
    taufbHist = np.zeros((2,len(t)),dtype=np.float64)
    fHist = np.zeros((2,len(t)),dtype=np.float64)
    fHHist = np.zeros((2,len(t)),dtype=np.float64)
    fDiffNormHist = np.zeros(len(t),dtype=np.float64)
    TCL = 0
    TCLindex = 0
    TCLfound = False
    TCLm = 0
    TCLmindex = 0
    TCLmfound = False

    #start save file
    savePath =  "/Users/huangzijing/Desktop"
    now = datetime.datetime.now()
    nownew = now.strftime("%Y-%m-%d-%H-%M-%S")
    path = savePath+"/sim-"+nownew
    os.mkdir(path)
    
    # loop through
    for jj in range(0,len(t)):
        # get the state and input data
        phidj,phiDdj,phiDDdj = dyn.getDesiredState(t[jj])
        phimj,phiDmj,phiDDmj,thetaHj = dyn.getState(t[jj])
        emj,eDmj,rmj = dyn.getErrorState(t[jj])
        tauj, tauffj, taufbj, _, _= dyn.getTau(t[jj], phi=phimj, phiD=phiDmj, thetaH=thetaHj)
        lamdaCLMinj,TCLj,_,_ = dyn.getCLstate()

        if not TCLfound:
            if TCLj > 0:
                TCL = TCLj
                TCLindex = jj
                TCLfound = True
        
        # save the data to the buffers
        phiHist[:,jj] = phiDmj
        phidHist[:,jj] = phiDdj
        eHist[:,jj] = emj
        rHist[:,jj] = rmj
        eNormHist[jj] = np.linalg.norm(emj)
        eNormHist[jj] = np.linalg.norm(rmj)
        thetaHHist[:,jj] = thetaHj
        lambdaCLMinHist[jj] = lamdaCLMinj
        tauHist[:,jj] = tauj
        tauffHist[:,jj] = tauffj
        taufbHist[:,jj] = taufbj


        if np.linalg.norm(phimj) > 5.0*np.linalg.norm(phidj) or np.linalg.norm(tauj) > 1000:
            print("GOING UNSTABLE")
            break
        
        # step the internal state of the dyanmics
        dyn.step(dt,t[jj])

    # plot the data
    #plot the angles
    phiplot,phiax = plot.subplots()
    phiax.plot(t,phidHist[0,:],color='darkviolet',linewidth=1,linestyle='-')
    phiax.plot(t,phiHist[0,:],color='darkviolet',linewidth=0.5,linestyle='--')
    phiax.plot(t,phidHist[1,:],color='limegreen',linewidth=1,linestyle='-')
    phiax.plot(t,phiHist[1,:],color='limegreen',linewidth=0.5,linestyle='--')
    phiax.set_xlabel("$t$ $(sec)$")
    phiax.set_ylabel("$\phi$")
    phiax.set_title("Angles")
    phiax.legend(["$\phi_{d1}$","$\phi_1$","$\phi_{d2}$","$\phi_2$"],loc='upper right')
    phiax.grid()
    phiplot.savefig(path+"/angle.pdf")

    #plot the error norm
    eNplot,eNax = plot.subplots()
    eNax.plot(t,eNormHist,color='darkviolet',linewidth=0.5,linestyle='-')
    eNax.set_xlabel("$t$ $(sec)$")
    eNax.set_ylabel("$\Vert e \Vert$")
    eNax.set_title("Error Norm RMS = "+str(np.around(np.sqrt(np.mean(eNormHist**2)),decimals=2)))
    eNax.grid()
    eNplot.savefig(path+"/errorNorm.pdf")

    #plot the inputs
    uplot,uax = plot.subplots()
    uax.plot(t,tauHist[0,:],color='red',linewidth=0.5,linestyle='-')
    uax.plot(t,tauffHist[0,:],color='green',linewidth=0.5,linestyle='-')
    uax.plot(t,taufbHist[0,:],color='blue',linewidth=0.5,linestyle='-')
    uax.plot(t,tauHist[1,:],color='red',linewidth=0.5,linestyle='--')
    uax.plot(t,tauffHist[1,:],color='green',linewidth=0.5,linestyle='--')
    uax.plot(t,taufbHist[1,:],color='blue',linewidth=0.5,linestyle='--')
    uax.set_xlabel("$t$ $(sec)$")
    uax.set_ylabel("$input$")
    uax.set_title("Control Input")
    uax.legend(["$\\tau_1$","$\\tau_{ff1}$","$\\tau_{fb1}$","$\\tau_2$","$\\tau_{ff2}$","$\\tau_{fb2}$"],loc='upper right')
    uax.grid()
    uplot.savefig(path+"/input.pdf")

    #plot the parameter estiamtes
    thHplot,thHax = plot.subplots()
    for ii in range(5):
        thHax.plot(t,thetaHHist[ii,:],color=np.random.rand(3),linewidth=1,linestyle='--')
    thHax.set_xlabel("$t$ $(sec)$")
    thHax.set_ylabel("$\theta_"+str(ii)+"$")
    thHax.set_title("Structured Parameter Estimates")
    thHax.grid()
    thHplot.savefig(path+"/thetaHat.pdf")


    


