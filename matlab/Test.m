function [ ] = Test(Case, pow)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
t = [50,100,150];
zf= [5,10];
format longe;
filename = 'ResultsTestInvPbNonoiseNonDec.xlsx';


if Case ==1
    sheet = 1;
    fd =@(z)(200-199.98*z);
elseif Case ==2
    fd=@(z)(200*((1-z).^pow));
    if pow == 1
       sheet = 2;
    elseif pow == 0.75
        sheet = 3;
    elseif pow == 0.5
        sheet = 4;
    elseif pow == 0.25
        sheet = 5;
    end
end

nt = length(t);
nz = length(zf);

    k1 = 5;
    kk1 = 8;
    for i = 1:nz
        for j = 1:nt
            zz = zf(i)
            tt = t(j)
         [Errs]=testinv3nonDecDifMeshfmincon(fd,0,zf(i),t(j),0,0,10^-6,0);

         start1 = num2str(k1,'%02d');
         end1 = num2str(kk1,'%02d');
         st1 = ['D' start1 ':K' end1 ]


         writematrix(Errs,filename,"Sheet",sheet,"Range",st1);

                  k1 = k1 + 9;

         kk1 = kk1 + 9;
        end
    end



    k1 = 5;
    kk1 = 8;
    for i = 1:nz
        for j = 1:nt
            zz = zf(i)
            tt = t(j)
         [Errs]=testinv3nonDecDifMeshfmincon(fd,0,zf(i),t(j),0,0,10^-8,0);
         %TestDirectPb1_NV2(fd,zf(i),t(j));D

         start1 = num2str(k1,'%02d');
         end1 = num2str(kk1,'%02d');
         st1 = ['N' start1 ':U' end1 ]


         writematrix(Errs,filename,'Sheet',sheet,'Range',st1);

                           k1 = k1 + 9;

         kk1 = kk1 + 9;
        end
    end

end

