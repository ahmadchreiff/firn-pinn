% this function generates the scalar V which takes as input
% z: vector of space partition points from 0 to 1, of length n
% h:  vector of length n-1 with entries the distance between consecutive z_i's
% D: vector of length n with entries D(i) diffusion coefficient of gas 
% CO2,air at position z(i)
% ralpha: a vector of length l that denotes to a specific gases cf*[1 2 3 ]
% Ug: n*(l) matrix with entries the concentration \rho_{\alpha}^o for the
% l gases given the diffusion coefficient Dag at end time T only
% v0: vector of length m with entries is \rho_{\alpha}^{atm}(j) i.e 
% value of \rho(1,j)
% M and K: n*n spareses matrices
% csts: a vector of length 9 with:
% csts(1) = n: number of meshing points for space, 
% csts(2) = m:  number of uniform time partition points 
% csts(3) = zf: last position in space before rescaling
% csts(4) = dt: the partition for time
% csts(5) =T: the end time
% csts(6) = Gf = G/f, 
% csts(7) =f1 = 1/f, 
% csts(8) = Maf = Ma/f= Ma*f1, 
% csts(9) = F
% outputs:
% V: the scalar which is the value of the objective function considering
% the the concentration of the l gases at end time
function [V] = InversePbRescV(D,z,h,ralpha,Ug,v0,M,K,csts)
n = csts(1);
m = csts(2);
l=length(ralpha);
%%%Dac=zeros(n,l); % Dac: n*l matrix with entries diffusion coefficient computed 
% using D_{CO2,air} and Dac(:,j) represents entries of diffusion coefficient 
% for alpha(j) 
[A,S]=COEFFv(h,n,D);
% % for j=1:l
% %     %for i=1:n
% %         Dac(:,j)=alpha(j)*D;
% %     %end
% % end 
Uc=zeros(n,l*m);% Uc: n*(l*m) matrix with entries the concentration \rho_{\alpha}^o
% for the l gases given the computed diffusion coefficient Dac
% % VV=zeros(n,l*m);
% % pUc = zeros(n,l*m);
% % pVV = zeros(n,l*m);
E=zeros(n,l); % E: n*l matrix with entries the error of the initial and computed 
% concentration for the l gases at the end time T
for j=1:l
    AA = ralpha(j)*A;
    SSS = ralpha(j)*S;
    Da = ralpha(j)*D;
    Uc(:,((j-1)*m+1):(j*m))=DirectPbResc(z,Da,v0,AA,SSS,M,K, csts);
    E(:,j)=Uc(:,j*m)-Ug(:,j);
%     VV(:,((j-1)*m+1):(j*m))=DirectPsi(m,alpha(j)*A,alpha(j)*S,M,K,n,zf,dt,T,Gf,f1,Maf,E(:,j));
end

% %%approximate the partial derivatives using finite differences 
% pUc(1,:) = (Uc(2,:) - Uc(1,:))/h(1);
% pUc(n,:) = (Uc(n,:) - Uc(n-1,:))/h(n-1);
% pUc(2:n-1,:) = (Uc(3:n,:) - Uc(1:n-2,:))/(2*h(2));%%Assuming uniform meshing
% 
% pVV(1,:) = (VV(2,:) - VV(1,:))/h(1);
% pVV(n,:) = (VV(n,:) - VV(n-1,:))/h(n-1);
% pVV(2:n-1,:) = (VV(3:n,:) - VV(1:n-2,:))/(2*h(2));%%Assuming uniform meshing
% 
% zeta = ((1/zf)*pUc-(Maf/f1)*Uc).*pVV;
% %wtilde = 2*sum(zeta,2);
% w = zeros(n,l);
% for j=1:l
%     w(:,j) = 2*sum(zeta(:,((j-1)*m+1):(j*m)),2)- zeta(:,(j-1)*m+1) - zeta(:,j*m);
%     w(:,j) = alpha(j)*w(:,j) ;
% end
% wtilde = sum(w,2) ;
% cnst = T*h(1)*dt*f1/zf;%%Assuming uniform meshing
% gradV = - cnst * wtilde;
% gradV(1) = 0.5*gradV(1);
% gradV(n) = 0.5*gradV(n);
% %N=zeros(1,l); % N: vector of length l with entries the square of the L_2 norm 
% % of the error E of each gas 
% % for j=1:l
% %     N(1,j)=norm(E(:,j)).^2;
% % end

V=sum(sum(E.*E)); % summation of all entries of N over all the gases


end