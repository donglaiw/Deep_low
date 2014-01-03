function y=U_roc(r,p)
[rr,id]=sort(r,'ascend');
% head and tail
% binary classification:
%y = 0.5*(rr(1)^2+(1-rr(end))^2);
y = 0;
%body
y = y + sum(0.5*(p(id(2:end))+p(id(1:end-1))).*(rr(2:end)-rr(1:end-1)));

