addpath('../util')
init
% extract commonly used options
if ~exist('Is','var')
    load data/test/dn_ucb
end
if ~exist('clusters','var')
    load clusters
end
addpath(genpath([VLIB '../Piotr/']));
psz = 35;
%{
pcluster = imPad(clusters.clusters,2,'symmetric');
%pcluster = clusters.clusters;
pcluster([1:2 end-1:end],[1:2 end-1:end],:) = 0;
pcluster = reshape(pcluster,psz^2,[]);
pcluster = [bsxfun(@rdivide,pcluster,sum(pcluster)) zeros(psz^2,1)];
rr2 = cell(1,numel(Is));
%}
rr = cell(1,numel(Is));

algo_id = 1; 
model_id = 1;
num_dim = [300 200 151];
num_epoch = 300;
str_dim = num2str(num_dim,'%d_');
str_dim(str_dim==' ')=[];
nn = [sprintf('result/%d_%d_',algo_id,model_id) str_dim sprintf('%d/dl_r1_',num_epoch)];

psz2 = 35;
parfor id = 1:numel(Is);
sz = size(Is{id});
if exist([nn num2str(id-1) '.mat'],'file')
    tmp=load([nn num2str(id-1)]) 
    [~,tmp_y] = max(tmp.result{1},[],2);
    rr{id} = reshape(tmp_y==151,sz([2 1])-psz+1)';
    %rr2{id} = scol2im(pcluster(:,tmp_y),psz,sz(2),sz(1),'average');
end
end
save(str_dim(1:end-1),'rr')
%save(str_dim(1:end-1),'rr','rr2')
%{
%}

