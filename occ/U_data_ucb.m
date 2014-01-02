DATA_UCB = '/home/Stephen/Desktop/Data/Seg/BSR/BSDS500/data/';
tstImgDir = [DATA_UCB '/images/test/'];
tstGtDir = [DATA_UCB '/groundTruth/test/'];
imgIds=dir([tstImgDir '*.jpg']);

imgIds={imgIds.name};
nImgs=length(imgIds);

addpath(genpath([VLIB '../Piotr/']));
radius=17;
Is =cell(1,nImgs);
gts =cell(1,nImgs);
for i = 1:nImgs
    I = imread([tstImgDir imgIds{i}]);    
    Is{i} = imPad(I,radius,'symmetric');
    gt=load([tstGtDir imgIds{i}(1:end-3) 'mat']);
    gt=gt.groundTruth;
    nGt=length(gt);
    gt_m = single(gt{1}.Boundaries);
    for j=2:nGt
        gt_m = gt_m+ single(gt{j}.Boundaries);
    end
    gts{i} = gt_m/nGt;
end

save dn_ucb Is gts

 
