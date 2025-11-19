% this function generates the matrix V which takes as 
% input:
% z: vector of space partition points from 0 to 1, of length n
% h:  vector of length n-1 with entries the distance between consecutive
% $z_i$'s
% Da: vector of length n with entries diffusion coefficient Da(i) 
% at position z(i)
% v0: vector of length m with entries is $\rho_{\alpha}^{atm}(j)$ 
% i.e value of $\rho(1,j)$
% M and K: n*n sparse matrices
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
% V: n*m matrix with entries the concentration  $\rho_{\alpha}^o$ at all
% positions of z(i) and t(j)
function[V]=DirectPbResc(z,Da,v0,A,S,M,K,csts)
% rescdirectpb = 1
n = csts(1);
m = csts(2);
zf = csts(3);
dt = csts(4);
T = csts(5);
Gf = csts(6);
f1 = csts(7);
Maf = csts(8);
F = csts(9);

dda = Da(1)+Da(2);
v3=1/6*Gf*z(2)-dda/(2*z(2))*f1/(zf^2)-dda*Maf/(4*zf)-F/(2*zf);
v1=(z(2)/6);
B=zeros(n,n); B(n,n)=F/zf;
S=(f1/(zf^2))*S;
A=(Maf/zf)*A;
K = (1/zf)*K;
VNK=2:n;
V=zeros(n,m);
V(1,:)=v0;

% % % hh = h(1);
% % % nnn = n;
% % % cccc = exp(1/(nnn^2*hh^2));
% % % %adding a nonzero rho^bar
% % % for jj = 1:nnn-2
% % %  V(jj+1,1)   = cccc*exp(-1/((nnn^2 - jj^2)*hh^2));
% % % end

% % %%adding a nonzero rho^bar
% % V(2,1) = V(1,1)/2;
% % V(3,1) = V(2,1)/2;
% % V(4,1) = V(3,1)/2;
% % V(5,1) = V(4,1)/2;
% % V(6,1) = V(5,1)/2;
% % 
% figure
%     plot(z,V(:,1));
%     pause;
%V1=zeros(n-1,1);
%V3=zeros(n-1,1); 
AG=M+T*dt*(Gf*M+S-K-A+B);
AA=AG(VNK,VNK);
[L,U,P]=lu(AA);
 M=M(VNK,VNK);
 
 vv1 = v1*(v0(2:m)-v0(1:m-1));
 vv3 = v3*v0(2:m);
for j=1:m-1
 %  V1(1,1)=vv1(j);%(v0(j+1)-v0(j))*v1;
 %  V3(1,1)=vv3(j);%(v0(j+1))*v3;
  %  V1(1,1)=(v0(j+1)-v0(j))*v1;
  %  V3(1,1)=(v0(j+1))*v3;
    RHS=M*V(VNK,j);%-V1-T*dt*V3;
    RHS(1) = RHS(1) - vv1(j) - T*dt*vv3(j);
    RHS = P*RHS;
    V(VNK,j+1)=U\(L\RHS);
%     jjn = max(V(VNK,j+1))
%     yyn = min(V(VNK,j+1))
%     figure
%     plot(z,V(:,j+1));
%     pause;
end
end