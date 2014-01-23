function [err2, err] = U_eval(ll,mat_x,mat_y,dotanh)
if iscell(ll)
    param = ll;
else
    load(['init_p' num2str(ll)],'param');
end
if ~exist('dotanh','var')
    dotanh = 1;
end
num_l = size(param,1);

for j=1:num_l-1
    pp = bsxfun(@plus,mat_x*param{j,1},param{j,2});
    if dotanh==0
        mat_x = sigmf(pp,[1 0]);
    else
        mat_x = tanh(pp);
    end
end

yhat = bsxfun(@plus,mat_x *param{end,1}, param{end,2});
err = (yhat - mat_y).^2; 
% training error: 0.9170 per patch
err2 = mean(sum(err,2));

