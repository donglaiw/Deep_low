addpath('../util');init
if ~exist('did','var')
 assert('need variable : did')
end


switch did
case 0
tstImgDir = [DATA_UCB '/images/train/'];
tstGtDir = [DATA_UCB '/groundTruth/train/'];
case 1
tstImgDir = [DATA_UCB '/images/val/'];
tstGtDir = [DATA_UCB '/groundTruth/val/'];
case 2
tstImgDir = [DATA_UCB '/images/test/'];
tstGtDir = [DATA_UCB '/groundTruth/test/'];
end

imgIds=dir([tstImgDir '*.jpg']);
imgIds={imgIds.name};
nImgs=length(imgIds);

Is =cell(1,nImgs);
Is2 =cell(1,nImgs);
gts =cell(1,nImgs);
segs =cell(1,nImgs);
for i = 1:nImgs
    Is{i} = imread([tstImgDir imgIds{i}]);    
    gt=load([tstGtDir imgIds{i}(1:end-3) 'mat']);
    gt=gt.groundTruth;
    nGt=length(gt);
    
    gts{i} = cell(1,nGt);
    for j=1:nGt
        gts{i}{j} = gt{j}.Boundaries;
    end
    segs{i} = cell(1,nGt);
    for j=1:nGt
        segs{i}{j} = gt{j}.Segmentation;
    end    
end

save(['dn_ucb' num2str(did)],'Is','segs','gts')


%{
load('dn_ucb1','gts')
sss=zeros(1,numel(gts));
eee=zeros(1,numel(gts));
parfor i=1:numel(gts)
sss(i) = numel(gts{i}{1});
eee(i) = nnz(sum(cat(3,gts{i}{:}),3));
end

sum(eee)/sum(sss)
%}

%{
load(['data/test/dn_ucb' num2str(did)])
parfor i=1:numel(Is)
    Is2{i} = rgb2gray(Is2{i});
    Is{i} = rgb2gray(Is{i});
end
save(['data/test/dn_ucbg' num2str(did)],'Is','Is2','gts')
%} 
