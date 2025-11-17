% this function generates the matrices M and K of the system which takes as 
% input: 
% h: vector of length n-1 with entries the distance between consecutifs  $z_i$'s
% (h(i-1)=z(i)-z(i-1))
% n: number of meshing points and n-1 intervals 
% F: constant  needed in the generation of matrix K 
% outputs: 
% M and K: n*n sparse matrices (without taking into consideration that
% $z_1$ is given)
function [M,K]=COEFFc(h,n,F)
I=zeros(3*n-4,1);
J=zeros(3*n-4,1);
MIJ=zeros(3*n-4,1);
KIJ=zeros(3*n-4,1);
pt=0;
for i=2:n-1
    pt=pt+1;
    I(pt)=i; J(pt)=i-1;
    MIJ(pt)=h(i-1)/6; 
    KIJ(pt)=-F/2;
    pt=pt+1;
    I(pt)=i; J(pt)=i;
    MIJ(pt)=(h(i-1)+h(i))/3;
    pt=pt+1;
    I(pt)=i; J(pt)=i+1;
    MIJ(pt)=h(i)/6; 
    KIJ(pt)=F/2;
end
pt=pt+1;
I(pt)=n; J(pt)=n-1;
MIJ(pt)=h(n-1)/6;
KIJ(pt)=-F/2;
pt=pt+1;
I(pt)=n; J(pt)=n;
MIJ(pt)=h(n-1)/3; 
KIJ(pt)=F/2;
M=sparse(I,J,MIJ);
K=sparse(I,J,KIJ);
end