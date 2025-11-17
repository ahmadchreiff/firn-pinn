function l =LinearSpline(x,y,X)
%Input: two equally sized column vectors x and y of length k
%          One vector X of length n
%Output: l = s(X) based on linear spline interpolation formula on the data (x,y)

n=length(X);
nn = length(x);X
ssx = size(x)
ssX = size(X)
ind = zeros(n,1);
% seek i : x(i) < r < x(i+1)
for i = 1:n
  ind(i) = find(x<=X(i), 1, 'last' );
end
% compute l=s(r)=s_i(r)
l = y(ind) + (y(ind+1)-y(ind)) ./ (x(ind+1)-x(ind)) .* (X-x(ind));
%size(l)