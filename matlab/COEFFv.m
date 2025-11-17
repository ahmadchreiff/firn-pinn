% this function generates the matrices A and S of the system which takes as
% input:
% h: vector of length n-1 with entries the distance between consecutifs  $z_i$'s
% (h(i-1)=z(i)-z(i-1))
% n: nomber of meshing points
% Da: vector of length n with entries diffusion coefficient Da(i) at 
% position z(i)
% outputs:
% A and S: n*n sparse matrices (without taking into consideration that
%  $z_1$ is given)
function [A,S]=COEFFv(h,n,Da)
I=zeros(3*n-4,1);
J=zeros(3*n-4,1);
SIJ=zeros(3*n-4,1);
AIJ=zeros(3*n-4,1);
pt=0;
for i=2:n-1
    pt=pt+1;
    I(pt)=i; J(pt)=i-1;
    AIJ(pt)=(-Da(i)-Da(i-1))/4;
    SIJ(pt)=(-Da(i)-Da(i-1))/(2*h(i-1));
    pt=pt+1;
    I(pt)=i; J(pt)=i;
    AIJ(pt)=(Da(i-1)-Da(i+1))/4;
    SIJ(pt)= (Da(i-1)+Da(i))/(2*h(i-1))+(Da(i+1)+Da(i))/(2*h(i));
    pt=pt+1;
    I(pt)=i; J(pt)=i+1;
    AIJ(pt)=(Da(i)+Da(i+1))/4;
    SIJ(pt)=(-Da(i)-Da(i+1))/(2*h(i));
end
pt=pt+1;
I(pt)=n; J(pt)=n-1;
AIJ(pt)=(-Da(n-1)-Da(n))/4;SIJ(pt)=(-Da(n)-Da(n-1))/(2*h(n-1));
pt=pt+1;
I(pt)=n; J(pt)=n;
AIJ(pt)=(Da(n-1)+Da(n))/4;SIJ(pt)=(Da(n)+Da(n-1))/(2*h(n-1));
A=sparse(I,J,AIJ);
S=sparse(I,J,SIJ);
end