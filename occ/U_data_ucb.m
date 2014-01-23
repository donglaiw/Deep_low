addpath('../util');init
if ~exist('did','var')
 assert('need variable : did')
end
switch did
case 2
tstImgDir = [DATA_UCB '/images/test/'];
tstGtDir = [DATA_UCB '/groundTruth/test/'];
case 1
tstImgDir = [DATA_UCB '/images/train/'];
tstGtDir = [DATA_UCB '/groundTruth/train/'];
end

imgIds=dir([tstImgDir '*.jpg']);

imgIds={imgIds.name};
nImgs=length(imgIds);

addpath(genpath([VLIB '../Piotr/']));
radius=17;
Is =cell(1,nImgs);
Is2 =cell(1,nImgs);
gts =cell(1,nImgs);
for i = 1:nImgs
    Is{i} = imread([tstImgDir imgIds{i}]);    
    Is2{i} = imPad(Is{i},radius,'symmetric');
    gt=load([tstGtDir imgIds{i}(1:end-3) 'mat']);
    gt=gt.groundTruth;
    nGt=length(gt);
    gts{i} = cell(1,nGt);
    for j=1:nGt
        gts{i}{j} = gt{j}.Boundaries;
    end
end

save(['dn_ucb' num2str(did)],'Is','Is2','gts')

%{
load(['data/test/dn_ucb' num2str(did)])
parfor i=1:numel(Is)
    Is2{i} = rgb2gray(Is2{i});
    Is{i} = rgb2gray(Is{i});
end
save(['data/test/dn_ucbg' num2str(did)],'Is','Is2','gts')
%} 
