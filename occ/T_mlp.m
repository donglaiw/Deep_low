% train_id=10;test_id=2;im_id=1;mlp_id=3;dim_n=2;T_mlp;imagesc(pb0)
addpath('../util');init
addpath(genpath([VLIB '../Piotr']))
addpath([VLIB 'Mid/Boundary/SketchTokens'])
% extract commonly used options
if ~exist('train_id','var');test_id=10;end
if ~exist('test_id','var');test_id=1;end
if ~exist('im_id','var');im_id=-1;end
if ~exist('mlp_id','var');mlp_id=1;end
if ~exist('dim_n','var');dim_n=2;end
%{
load(['data/test/dn_ucb' num2str(test_id)])
load('st_bd')
%}
psz = 35;
%{
if ~exist('clusters','var')
    load clusters
end
pcluster = imPad(clusters.clusters,2,'symmetric');
%pcluster = clusters.clusters;
pcluster([1:2 end-1:end],[1:2 end-1:end],:) = 0;
pcluster = reshape(pcluster,psz^2,[]);
pcluster = [bsxfun(@rdivide,pcluster,sum(pcluster)) zeros(psz^2,1)];
%}
num = numel(Is);
%rr = cell(1,num);
pb = cell(1,num);

algo_id = 1;
switch mlp_id
case 1
model_id = 0;
num_dim = [500 dim_n];
num_epoch = 100;
case 2
model_id = 1;
num_dim = [500 500 dim_n];
num_epoch = 100;
case 3
model_id = 2;
num_dim = [500 500 500 dim_n];
num_epoch = 100;
end

str_dim = num2str(num_dim,'%d_');
str_dim(str_dim==' ')=[];
nn = [sprintf('result/%d_%d_',algo_id,model_id) str_dim sprintf('%d_%d/dl_r%d_',train_id,num_epoch,test_id)];


gt_b=sum(cat(3,gts{im_id}{:}),3);
gt_id = gt_b~=0;
sz = size(Is{im_id});
if mlp_id<=0
    if ~exist('sts','var')
    end
    pre_y = (sts{im_id}==151)+1;
else
    if exist([nn num2str(im_id-1) '.mat'],'file')
        %python P_occ.py 1 2 100 500,500,500,151 9 2
        tmp=load([nn num2str(im_id-1)]); 
        [~,pred_y] = max(tmp.result{1},[],2);
        pre_y = reshape(pred_y,sz([2 1]))';
        pb0 = stToEdges( reshape(tmp.result{1},[sz([2 1]) num_dim(end)]), 1 )';
    else
        disp(['no exist'])
    end
end
err = (nnz(pre_y(gt_id)==2)+nnz(pre_y==1)-nnz(pre_y(gt_id)==1))/ prod(sz(1:2)) 

