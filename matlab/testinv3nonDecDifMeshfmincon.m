%This function test the inverse problem using fmincon with 2 different
% iterative solvers, sqp and interior point method, for different mesh sizes.
% It uses the rescaled direct problem

% It takes as input:
%   fd:     exact Dco2
%   Decr:   the type of minimization problem as decreasing or non
%           decreasing. Decr == 0 => non decreasing, else decreasing
%   zF:     the length of the space interval
%   T:      the end time at which we assume we have the data and want to
%           use it to extract the D_CO2
%   Un:     an integer specifying the type of meshing. If un == 0 then we
%           set uniform meshing with mesh size h that belongs to [1/16,
%           1/32, 1/64, 1/128, 1/256].  If un == 1 then we set a nonuniform
%           meshing [0:h/16:0.0625:h/8:0.125:h/4:0.25:h/2:0.5:h:1]
%   SS:     an integer specifying the type of initial guess for fmincon. If
%           SS==0 then the initial guess is set to D = zeros, else it is set
%           to a random vector.
%   tolfun: tolerance set to fmincon. Default tolerance is 10^-6
%   X:      an integer specifying the type of added noise to the generated
%           data. If X == 0, then no noise is added. If X == 1, additive
%           noise is used. Else the white gaussian noise is added.
%   sgnp:   If X~=0, then sgnp refers to the signal noise power which
%           varies between 10^-3 and 10^-6
%  
%   noise:  refers to the noise percentage that varies between 0% and 20%

%It outputs:
%   Errs:   A matrix of size 5x12, with runtime, iterations, 4 L2 errors,
%           for SQP method and IP method, for the 5 h values

function[Errs]=testinv3nonDecDifMeshfmincon(fd,Decr,zF,T,Un,SS,tolfun,X,sgnp,noise)
format longe;
%the constants:
f=0.2;f1=1/f;
Maf=f1*(0.04*9.8)/(8.314*260);
G=10+0.03;
Gf=f1*G;
F=200+485;

jmax = 1000;


csts(3) = zF;
csts(5) = T ;
csts(6) = Gf ;
 csts(7) = f1;
csts(8) = Maf;
csts(9) = F;

% the rescaled domain space [a,b]=[0,1]:
a=0;b=1;

%% setting the uniform or nonuniform space mesh
% % if Un==0
 %  r = 1/32;
r=[1/8,1/16, 1/32, 1/64];%, 1/128];%, 1/256]; 
    %r = [1/16, 1/32, 1/64, 1/128];]
   % r = 1/32;
    %p=L/r(e)+1;% the largest number of mesh points corresponding to the finest h value.
% % else
% %     r=[1/4, 1/8, 1/16, 1/32, 1/64]; 
% % end
e=length(r);
%e = 1
Rtime = zeros(e,2);
options1 = optimoptions('fmincon', ...
    'Algorithm','sqp',...
    'TolFun',tolfun,...
    'MaxIter',10000,...
    'MaxFunEvals',300000); % input for fmincon
options2 = optimoptions('fmincon', ...
    'TolFun',tolfun,...
    'MaxIter',10000,...
    'MaxFunEvals',300000); % input for fmincon
err=zeros(2,e); % relative L_2 error between obtained D and the given Da
Err=zeros(2,e); % absolutebL_2 error between obtained D and the given Da

cf=0.5; %constant
ralpha=cf*[1,2,3]; %the alpha gases
%ralpha =1;
l=length(ralpha);

if X~=0
    snr=100/noise;% signal to noise ratio = 1/noise 
end

vd = @(t)(2*(T*t).^(0.25));

%fd=@(z)(200-199.98*z); 
for o=1:e
%%------------------------------------------------------------------------------------------------------------------------------------------------------
%% generate data at a different mesh
H = 1/(1+1/r(o));
dt = H;  
tg = 0:dt:1; %vector of all time partition points
zg = 0:H:1;

    %z = 0:H:1; %vector of all space partition points
    ng=length(zg); % n: number of meshing points in space
    m=length(tg); % m: number of meshing points in time
     csts(1) = ng ;
     csts(2) = m ;
     csts(4) = dt ;

    h=zeros(ng-1,1);
     %h: vector of length n-1 with entries the distance between
     % consecutifs z_i's needed in generating the four matrices A,S,M,K
    h(1:ng-1)=zg(2:ng)-zg(1:ng-1); 
    [M,K]=COEFFc(h,ng,F); % generating matrices M and K

%     v0=zeros(1,m); % v0: vector of length m with entries is rho_{\alpha}^{atm}(j)
%     % i.e value of $\rho(1,j)  
     v0 = vd(tg);
     
    %Da=zeros(n,1);
    % Da: vector of length n with entries diffusion coefficient, i.e. the Da(i) at position z(i),
    % Da is the exact D_CO2 we are considering
    
     Dazg=(fd(zg))';
     [A,S]=COEFFv(h,ng,Dazg); % generating matrices M and K 

   Dag = Dazg*ralpha;
    sDag = size(Dag) 
    %pause
    Ug=zeros(ng,l*m);% Ug: n*(l*m) matrix with entries the concentration
    %\rho_{\alpha}^o for the l gases given the diffusion coefficient Dag
    Ugl=zeros(ng,l);% the vector where we extract the solution at the endtime for each of the l gases

  %  DirectPbResc(z,m,h,Da,v0,M,K,n,zf,dt,T,Gf,f1,Maf,F)
    %% Generating the exact data by calling the direct Problem
    for j=1:l
        Ug(:,((j-1)*m+1):(j*m))=DirectPbResc(zg,Dag(:,j),v0,ralpha(j)*A,ralpha(j)*S,M,K,csts);
        Ugl(:,j) =  Ug(:,(j*m));
    end
    
%%_------------------------------------------------------------------------------------------------------------------------------------------------------------------------
%%   starting the iteration
%%_------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    H=r(o); % H: uniform partition size
    dt=H; % dt: partition for time
    t = 0:dt:1; %vector of all time partition points
 if Un==0
     z = 0:H:1;
 else
    zzz0 = 0:H/16:0.0625; 
    zz0 = 0.0625+H/8:H/8:0.125; 
    zz1 = 0.125+H/4:H/4:0.25; 
    zz2 = 0.25+H/2:H/2:0.5;
    zz3 = 0.5+H:H:1;
    z = [zzz0, zz0, zz1, zz2, zz3];
 end
    %z = 0:H:1; %vector of all space partition points
    n=length(z); % n: number of meshing points in space
    m=length(t); % m: number of meshing points in time
     csts(1) = n ;
     csts(2) = m ;
     csts(4) = dt ;
    x0=200*rand(n,1);

    h=zeros(n-1,1);
     %h: vector of length n-1 with entries the distance between
     % consecutifs z_i's needed in generating the four matrices A,S,M,K
    h(1:n-1)=z(2:n)-z(1:n-1); 
    [M,K]=COEFFc(h,n,F); % generating matrices M and K

%     v0=zeros(1,m); % v0: vector of length m with entries is rho_{\alpha}^{atm}(j)
%     % i.e value of $\rho(1,j)  
     v0 = vd(t);
     
    %Da=zeros(n,1);
    % Da: vector of length n with entries diffusion coefficient, i.e. the Da(i) at position z(i),
    % Da is the exact D_CO2 we are considering
    
     Da=(fd(z))';
    % [A,S]=COEFFv(h,n,Da); % generating matrices M and K 


     Ugli=zeros(n,l);% the vector where we extract the solution at the endtime for each of the l gases
    Uu=zeros(n,l);% the vector where we add noise to Ugl

  %  DirectPbResc(z,m,h,Da,v0,M,K,n,zf,dt,T,Gf,f1,Maf,F)
    %% Interpolating the exact data to the given mesh
    for j=1:l
        %Ug(:,((j-1)*m+1):(j*m))=DirectPbResc(z,Dag(:,j),v0,ralpha(j)*A,ralpha(j)*S,M,K,csts);
        %Ugl(:,j) =  Ug(:,(j*m));
        Ugli(2:n-1,j) = LinearSpline(zg',Ugl(:,j),z(2:n-1)');
        Ugli(n,j) = Ugl(ng,j) ;        
        Ugli(1,j) = Ugl(1,j) ;
       % figure;
       % plot(zg,Ugl(:,j),'k', z,Ugli(:,j),'b');   
    end
    %% Adding noise to the exact Data
    if(X==1)
         for j=1:l
             %maxU = max(abs(Ugl(:,j)))
            Uu(:,j)= Ugli(:,j)+ (sgnp.*Ugli(:,j));
         end
    elseif(X==0)
        for j=1:l
           Uu(:,j)= Ugli(:,j);
        end
    else
        for j=1:l
           Uu(:,j)= awgn(Ugli(:,j),snr,sgnp,'linear');
        end
    end
    ddd = Uu-Ugli; %error added to Ugl
    relErrNoise = sqrt(sum(ddd.*ddd))./sqrt(sum(Ugli.*Ugli)) % relative error between exact solution and the soltion with noise.;

    %% Setting the initial guess and input for fmincon
    % D0: initial D_{CO2,air} which is a vector of length n
    % D0=zeros(n,1); %for sections \ref{sec:4.3.1} and \ref{sec:4.3.3}
    if(SS~=0)
        SS = SS
        D0=x0;
    else
        D0 = zeros(n,1);
        %D0=100*ones(n,1);
    end

% fmincon settings
   Aeq=[]; beq=[]; 
    nonlcon=[];lb=zeros(n,1);ub=[];% inputs for fmincon
   dddd = ones(n-1,1);
   ddd = ones(n,1);

   if Decr == 0%Non decreasing optimization pb
       AA = [];
       bb = [];
   else%Decreasing optimization pb
     bb=zeros(n,1) ;
     AA= -diag(ddd)+diag(dddd,1);
   end
      
   % input for fmincon: lower bound for D(i) (D(i) \geq  0)
    %% Calling fmincon with the two options defined before
    % finding D_{CO2,air} by minimizing the objective function created "InversePbV":
    FFUN = @(DDD)( InversePbRescV(DDD,z,h,ralpha,Uu,v0,M,K,csts));
  

        tic;
    [D1,fval1,exitflag1,output1]=fmincon(FFUN,D0,AA,bb,Aeq,beq,lb,ub,nonlcon,options1);
    Rtime(o,1) = toc

         fval1 = fval1
        iter = output1.iterations
         exitflag1 = exitflag1
        % figure;
        % plot(zg,Dazg,'b',z,D1,'r')
        % legend('D_g','D_c');
        % xlabel('Rescaled z','fontweight','bold','fontsize',14);
        % tit1 = ['SQP, T=' num2str(T,'%01d') ', z_F=' num2str(zF,'%02d')  ', h=1/' num2str(1/r(o),'%01d')]; 
        % title(tit1)
        %saveas(gcf, tit1, 'fig');


         % checking the obtained rho solution
         % % % [A,S]=COEFFv(h,n,D1); % generating matrices M and K 
         % % % Ugs(:,(1):(m))=DirectPbResc(zg,Dag(:,j),v0,ralpha(j)*A,ralpha(j)*S,M,K,csts);
         % % % figure;
         % % % plot(z,Ugs(:,(m)),'b')
    
    fprintf('interior point method');% for h=%d', H, '\n')

    tic;
    [D2,fval2,exitflag2,output2]=fmincon(FFUN,D0,AA,bb,Aeq,beq,lb,ub,nonlcon,options2);
    Rtime(o,2) = toc

    
    D=zeros(n,2);
    D(:,1)=D1;
    D(:,2)=D2;
        
    iter = output2.iterations
    fval2 = fval2
    output2 = output2
    exitflag2 = exitflag2
     % figure;
     % plot(z,Da,'b',z,D2,'r');
     % legend('D_g','D_c');
     % tit2 = ['IP, T=' num2str(T,'%01d') ', z_F=' num2str(zF,'%02d')  ', h=1/' num2str(1/r(o),'%01d')];
     %  xlabel('Rescaled z','fontweight','bold','fontsize',14);
     % title(tit2); 
  %   ylim( [0, 200]);
     
       % % % [A,S]=COEFFv(h,n,D2); % generating matrices M and K 
       % % %  Ugs(:,(1):(m))=DirectPbResc(zg,Dag(:,j),v0,ralpha(j)*A,ralpha(j)*S,M,K,csts);
       % % %  figure;
       % % %  plot(z,Ugs(:,(m)),'b')

     %saveas(gcf,'/MATLAB Drive/Firn-Rescaled/InversePb/tit2.jpg')
     %saveas(gcf, tit2, 'fig');
    %% Computing the absolute and relative errors
      nDa=norm(Da);
    for i=1:2%changed from 2 to 1
        v=D(:,i)-Da;
        Err(i,o)=norm(v);
     %   Errav(i,o)=Err(i,o)/n;
        err(i,o)=Err(i,o)/nDa;
   %     errav(i,o) = err(i,o)/n;

        Errs(o,3+(i-1)*4) =   Err(i,o);
     %   Errs(o,4+(i-1)*6) =   Errav(i,o);
        Errs(o,4+(i-1)*4) =   err(i,o);
      %  Errs(o,6+(i-1)*6) =   errav(i,o);
    end 
     Errs(o,2) =   output1.iterations;
     Errs(o,6) =   output2.iterations;

end

 Errs(:,1) = Rtime(:,1);
 Errs(:,5) = Rtime(:,2);
% 
 Errs = Errs

% Errav = Errav
% Err = Err
% relErrD = err
% figure;
% plot(r,err(1,:),'r*')
% figure;
% plot(r(1:e-1),err(1,1:e-1),'r*')
% figure;
% plot(r,err(2,:),'r*')
% figure;
% plot(r(1:e-1),err(2,1:e-1),'r*')