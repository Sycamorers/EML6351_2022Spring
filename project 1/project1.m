clc
clear

for j = 1:1
    %time step
    ts = 0.005;
    t(1) = 0;
    tf = 60;
    S = tf/ts;

    %gains
    alpha = 15;
    beta = 15;
    Gamma = 5;
    
    %parameters
    m1 = 2;
    m2 = 2;
    l1 = 0.5;
    l2 = 0.5;
    mupper = 3;
    mlower = 1;
    lupper = 0.75;
    llower = 0.25;
    g = 9.8;
    
    %desired trajectory parameters
    phid1 = pi/8;
    phid2 = pi/4;
    phid = [phid1;phid2];
    fphid = 0.2;
    aphid = pi/2;
    bphid1 = pi/2;
    bphid2 = pi/4;
    bphid = [bphid1;bphid2];
    
    % initialize theta
    theta(:,1) = [(m1+m2)*(l1^2)+m2*(l2^2);
                   m2*l1*l2;
                   m2*(l2^2);
                   (m1+m2)*l1;
                   m2*l2];
               
    % initialize theta estimate to the lowerbounds
    thetaH(:,1) = [(mlower+mlower)*llower^2+mlower*llower^2;
                   mlower*llower*llower;
                   mlower*llower^2;
                   (mlower+mlower)*llower;
                   mlower*llower];
      
    % initialize state
    phi(:,1) = [phid1*sin(-aphid)-bphid1; 
                phid2*sin(-aphid)+bphid2];
    phiD(:,1) = [0;0];
    phiDD(:,1) = [0;0];

    for i = 1:S
        
        % desired trajectory
        phid(:,i) = [phid1*sin(2*pi*fphid*t(i)-aphid)-bphid1;
                     phid2*sin(2*pi*fphid*t(i)-aphid)+bphid2];
        phidD(:,i) = [2*pi*fphid*phid1*cos(2*pi*fphid*t(i)-aphid);
                      2*pi*fphid*phid2*cos(2*pi*fphid*t(i)-aphid)];
        phidDD(:,i) = [-(2*pi*fphid)^2*phid1*sin(2*pi*fphid*t(i)-aphid);
                       -(2*pi*fphid)^2*phid2*sin(2*pi*fphid*t(i)-aphid)];
        % tracking error
        %eDD(:,i) = phidDD(:,i)-phiDD(:,i);
        eD(:,i) = phidD(:,i)-phiD(:,i);
        e(:,i) = phid(:,i)-phi(:,i);
        
        % filtered tracking error
        %rD(:,i) = eDD(:,i) + alpha*eD(:,i);
        r(:,i) = eD(:,i)+alpha*e(:,i); 
        
        %c1 c2 s1 s2 c12 s12
        c1(i) = cos(phi(1,i));
        c2(i) = cos(phi(2,i));
        s1(i) = sin(phi(1,i));
        s2(i) = sin(phi(2,i));
        s12(i) = sin(phi(1,i)+phi(2,i));
        c12(i) = cos(phi(1,i)+phi(2,i));
        
        % inertia matrix
        M = [m1*(l1^2)+m2*(l2^2+l1^2+2*l1*l2*c2(i)), m2*(l1*l2*c2(i)+l2^2);
             m2*(l1*l2*c2(i)+l2^2), m2*l2^2];
         
        % centripetal coriolis matrix
        C = [-2*m2*l1*l2*sin(phi(2,i))*phiD(1,i)*phiD(2,i)-m2*l1*l2*sin(phi(2,i))*(phiD(2,i)^2);
             m2*l1*l2*sin(phi(2,i))*(phiD(1,i)^2)];
         
        % gravity matrix 
        G = [(m1+m2)*g*l1*c1(i)+m2*g*l2*c12(i);
             m2*g*l2*c12(i)];
         
        % intertia matrix regressor
        vphi(:,i) = phidDD(:,i) + alpha*eD(:,i);
        YM = [vphi(1,i), 2*c2(i)*vphi(1,i)+c2(i)*vphi(2,i), vphi(2,i), 0, 0;
              0, c2(i)*vphi(1,i), vphi(1,i)+vphi(2,i), 0, 0];
          
        % centripetal coriolis matrix regressor
        YC = [ 0, -(2*sin(phi(2,i))*phiD(1,i)*phiD(2,i)+sin(phi(2,i))*(phiD(2,i)^2)), 0, 0, 0;
              0, sin(phi(2,i))*(phiD(1,i)^2), 0, 0, 0];
          
        % gravity matrix regressor 
        YG = [0 0 0 g*c1(i) g*c12(i);
              0 0 0 0 g*c12(i)];
          
        % intertia matrix derivative regressor
        YMD = [0, -(2*sin(phi(2,i))*phiD(2,i)*r(1,i)+sin(phi(2,i))*phiD(2,i)*r(2,i)) 0, 0, 0;
               0, -sin(phi(2,i))*phiD(2,i)*r(1,i), 0, 0, 0];
           
        % total matrix regressor
        %Y = YM + YC + YG + 0.5 * YMD;
        Y = YM + YC + YG + YMD;
        
        % input
        tauff(:,i) = Y * thetaH(:,i);
        taufb(:,i) = e(:,i)+ beta*r(:,i);
        tau(:,i) =  tauff(:,i) + taufb(:,i);
        
        % acceleration
        phiDD(:,i) = M\(tau(:,i)- C - G);
        
        % estimation error
        thetaT(:,i) = theta(:,1) - thetaH(:,i);  
        
        % update law 
        thetaHD(:,i) = Gamma * Y' * r(:,i);
        
        %norm
        enorm(i) = vecnorm(e(:,i));
        rnorm(i) = vecnorm(r(:,i));
        taunorm(i) = vecnorm(tau(:,i));
        tauffnorm(i) = vecnorm(tauff(:,i));
        taufbnorm(i) = vecnorm(taufb(:,i));
        
        if i < S
        t(i+1) = t(i)+ts;
        phiD(:,i+1) = phiD(:,i)+ phiDD(:,i) * ts;
        phi(:,i+1) = phi(:,i)+ phiD(:,i) * ts;
        thetaH(:,i+1) =  thetaH(:,i) + ts * thetaHD(:,i);
        end    
    end     
end
figure(1)
plot(t,enorm)
legend('e')
title('e','FontSize',16)
figure(2)
plot(t,rnorm)
legend('r')
title('r','FontSize',16)
figure(3)
plot(t, taunorm, t, tauffnorm, t, taufbnorm)
legend('tau', 'tauff', 'taufb')
title('tau','FontSize',16)
figure(4)
plot(t,thetaT)
title('theta','FontSize',16)
figure(5)
plot(t,phid(1,:),t,phi(1,:))
legend('phid', 'phi')




