addpath('../../util');init
stream=RandStream('mrg32k3a','Seed',1);
set(stream,'Substream',1);
RandStream.setGlobalStream( stream );

did=0;
addpath('/home/Stephen/Desktop/Data/Seg/BSR')
load(['dn_ucb_pb' num2str(did)])
psz = 35;
psz_h = (1+psz)/2;
mat_x = cell(1,numel(gts_pb));
num_perimg = 150;
parfor i=1:numel(gts_pb)
	mat_x{i} = im2col(gts_pb{i},[psz psz]);
    tmp_gt = gts_pb{i}(psz_h:(end-psz_h+1),psz_h:(end-psz_h+1));   
    sz = size(tmp_gt);
    ind = reshape(1:prod(sz),sz);
    bd_thres = 0.6;
    ind2 = ind;ind2(tmp_gt(ind)<=bd_thres) = [];
    while numel(ind2)<num_perimg
        bd_thres = bd_thres-0.1;
        ind2 = ind;ind2(tmp_gt(ind)<=bd_thres) = [];
    end
	tmp_ind = randsample(ind2,num_perimg);
	mat_x{i} = mat_x{i}(:,tmp_ind);
end
mat_x = cell2mat(mat_x);
save(['spam' num2str(did)],'mat_x')


