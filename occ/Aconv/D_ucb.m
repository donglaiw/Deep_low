addpath('../../util'),init 

if ~exist('did','var')
   did = 0;
end
load([DATA_UCB 'dn_ucb' num2str(did)]) 
num_p = 1000;
num_im = numel(Is);
psz = 11;
psz_h = (1+psz)/2;
mat_x= zeros(3*psz^2,num_p*num_im,'single');
mat_y= zeros(1,num_p*num_im,'single');
cc= 0;
for i=1:num_im
    
    tmp = [im2col(Is{i}(:,:,1),[psz psz]);...
    im2col(Is{i}(:,:,2),[psz psz]);...
    im2col(Is{i}(:,:,3),[psz psz])];
    ind = randsample(size(tmp,2),num_p);
    mat_x(:,cc+(1:num_p)) = tmp(:,ind);

    gt_val = mean(single(cat(3,gts{i}{:})),3);
    gt_val = gt_val(psz_h:(end-psz_h+1),psz_h:(end-psz_h+1));
    mat_y(:,cc+(1:num_p)) = gt_val(ind);
    cc = cc+ num_p;
end
save(['conv_' num2str(psz) '_' num2str(did)],'mat_x','mat_y')

%{
psz=11;
for did = 0:1
load(['../data/train/conv_' num2str(psz) '_' num2str(did)],'mat_x','mat_y')
mat_x = single(mat_x)/255;
mat_y = single(mat_y);
save(['../data/train/conv_' num2str(psz) '_' num2str(did)],'mat_x','mat_y')
end
did=2;
load(['../data/test/conv_' num2str(psz) '_' num2str(did)],'mat_x','mat_y')
mat_x = single(mat_x)/255;
mat_y = single(mat_y);
save(['../data/test/conv_' num2str(psz) '_' num2str(did)],'mat_x','mat_y')



%}

