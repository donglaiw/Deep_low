% test_id=2;im_id=1;mlp_id=3;T_mlp;imagesc(pb0)
addpath('../util');init
addpath(genpath([VLIB '../Piotr']))
addpath([VLIB 'Mid/Boundary/SketchTokens'])
% extract commonly used options
if ~exist('test_id','var');test_id=1;end
if ~exist('im_id','var');im_id=-1;end
if ~exist('mlp_id','var');mlp_id=1;end

load(['data/test/dn_ucb' num2str(test_id)])
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
num_dim = [500 2];
num_epoch = 100;
case 2
model_id = 1;
num_dim = [300 200 2];
num_epoch = 100;
case 3
model_id = 2;
num_dim = [100 200 100 2];
num_epoch = 100;
end

str_dim = num2str(num_dim,'%d_');
str_dim(str_dim==' ')=[];
nn = [sprintf('result/%d_%d_',algo_id,model_id) str_dim sprintf('%d/dl_r%d_',num_epoch,test_id)];

if im_id<=0
parfor id = 1:num
sz = size(Is{id});
if exist([nn num2str(id-1) '.mat'],'file')
    tmp=load([nn num2str(id-1)]) 
    %[~,tmp_y] = max(tmp.result{1},[],2);
    %rr{id} = reshape(tmp_y==151,sz([2 1]))';
    pb{id} = stToEdges( reshape(tmp.result{1},[sz([2 1]) num_dim(end)]), 1 )';
    %rr2{id} = scol2im(pcluster(:,tmp_y),psz,sz(2),sz(1),'average');
end
end
%save(str_dim(1:end-1),'rr','pb')
save(str_dim(1:end-1),'pb')
else
    id = im_id;
    sz = size(Is{id});
    if exist([nn num2str(id-1) '.mat'],'file')
        tmp=load([nn num2str(id-1)]); 
        pb0 = stToEdges( reshape(tmp.result{1},[sz([2 1]) num_dim(end)]), 1 )';
    else
        disp(['no exist'])
    end

end

