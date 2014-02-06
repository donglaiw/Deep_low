addpath('../util');init
addpath(genpath([VLIB 'Mid/Boundary/SketchTokens']))
P_DIR = [VLIB '../Piotr'];
addpath(genpath(P_DIR))
opts={};
dfs={'nClusters',150, 'nTrees',25, 'radius',17, 'nPos',1000, 'nNeg',16000,...
    'negDist',2, 'minCount',4, 'nCells',5, 'normRad',5, 'normConst',0.01, ...
    'nOrients',[4 4 0], 'sigmas',[0 1.5 5], 'chnsSmooth',2, 'fracFtrs',1, ...
    'seed',1, 'modelDir','models/', 'modelFnm','model', ...
    'clusterFnm',[VLIB 'Mid/Boundary/SketchTokens/st_data/clusters.mat'], 'bsdsDir','/home/Stephen/Desktop/Data/Seg/BSR/BSDS500/data/'};
opts = getPrmDflt(opts,dfs,1);
if ~exist('did','var')
did = 1;
end


switch did
case -1
    % train
    opts.nPos=0;
    opts.nNeg=1500;
    trnImgDir = [opts.bsdsDir '/images/train/'];
    trnGtDir = [opts.bsdsDir '/groundTruth/train/'];
    case 0
    % train
    opts.nPos=1000;
    opts.nNeg=0;
    trnImgDir = [opts.bsdsDir '/images/train/'];
    trnGtDir = [opts.bsdsDir '/groundTruth/train/'];
    case 1
    % train
    opts.nPos=100;
    opts.nNeg=80;
    trnImgDir = [opts.bsdsDir '/images/train/'];
    trnGtDir = [opts.bsdsDir '/groundTruth/train/'];
case 2
    % test
    opts.nPos=100;
    opts.nNeg=160;
    trnImgDir = [opts.bsdsDir '/images/val/'];
    trnGtDir = [opts.bsdsDir '/groundTruth/val/'];
end

stream=RandStream('mrg32k3a','Seed',ceil(1e5*rand()));
% set global stream to stream with given substream (will undo at end)
streamOrig = RandStream.getGlobalStream();
RandStream.setGlobalStream( stream );
rand()


% if forest exists load it and return

% compute constants and store in opts
nTrees=opts.nTrees;
nCells=opts.nCells;

patchSiz=opts.radius*2+1;
opts.patchSiz=patchSiz;

nChns = size(stChns(ones(2,2,3),opts),3);
opts.nChns=nChns;

opts.nChnFtrs = patchSiz*patchSiz*nChns;
opts.nSimFtrs = (nCells*nCells)*(nCells*nCells-1)/2*nChns;
opts.nTotFtrs = opts.nChnFtrs + opts.nSimFtrs;
opts.cellRad = round(patchSiz/nCells/2);
tmp=opts.cellRad*2+1;
opts.cellStep = tmp-ceil((nCells*tmp-patchSiz)/(nCells-1)); disp(opts);
assert( (nCells == 0) || (mod(nCells,2)==1 && (nCells-1)*opts.cellStep+tmp <= patchSiz ));


imgIds=dir([trnImgDir '*.jpg']);
imgIds={imgIds.name};
nImgs=length(imgIds);
for i=1:nImgs,
    imgIds{i}=imgIds{i}(1:end-4);
end

% extract commonly used options
radius=opts.radius;
patchSiz=opts.patchSiz;
nChns=opts.nChns;
nTotFtrs=opts.nTotFtrs;
nClusters=opts.nClusters;
nPos=opts.nPos;
nNeg=opts.nNeg;

% finalize setup

% sample nPos positive patch locations per cluster
clstr=load(opts.clusterFnm);
clstr=clstr.clusters;
for i = 1:nClusters
    if i==1
        centers=[];
    end
    ids = find(clstr.clusterId == i);
    ids = ids(randperm(length(ids),min(nPos,length(ids))));
    centers = [centers; [clstr.x(ids),clstr.y(ids),clstr.imId(ids),...
        clstr.clusterId(ids),clstr.gtId(ids)]]; %#ok<AGROW>
end
%{
    i=1;
    ids = find(clstr.imId == i);
    plot(clstr.x(ids),clstr.y(ids),'b.')

%}
% collect positive and negative patches and compute features
fids=sort(randperm(nTotFtrs,round(nTotFtrs*opts.fracFtrs)));
k = size(centers,1)+nNeg*nImgs;
ftrs = zeros(k,length(fids),'single');
ftrs_bd = zeros(k,patchSiz^2,'single');
ftrs_rgb = zeros(k,patchSiz^2*3,'single');
labels = zeros(k,1); k = 0;
tid = ticStatus('Collecting data',1,1);
for i = 1:nImgs
    % get image and compute channels
    gt=load([trnGtDir imgIds{i} '.mat']);
    gt=gt.groundTruth;
    
    I = imread([trnImgDir imgIds{i} '.jpg']);
    I = imPad(I,radius,'symmetric');
    chns = stChns(I,opts);
    
    % sample positive patch locations
    centers1=centers(centers(:,3)==i,:);
    lbls1=centers1(:,4);
    xy1=single(centers1(:,[1 2]));
    
    % sample negative patch locations
    M=false(size(I,1)-2*radius,size(I,2)-2*radius);
    nGt=length(gt);
    for j=1:nGt
        M1=gt{j}.Boundaries;
        if ~isempty(M1)
            M=M | M1;
        end
    end
    M(bwdist(M)<opts.negDist)=1;
    M=~M;
    M([1:radius end-radius:end],:)=0;
    M(:,[1:radius end-radius:end])=0;
    [y,x]=find(M);
    k1=min(length(y),nNeg);
    
    rp=randperm(length(y),k1);
    y=y(rp);
    x=x(rp);
    xy0=[x y];
    lbls0=ones(k1,1)*(nClusters+1);
    
    % crop patches
    xy=[xy1; xy0];
    lbls=[lbls1; lbls0];
    k1=length(lbls);
    ps=zeros(patchSiz,patchSiz,nChns,k1,'single');
    gt_m = single(gt{1}.Boundaries);
    for j=2:nGt
        gt_m = gt_m+ single(gt{j}.Boundaries);
    end
    gt_m = imPad(gt_m/nGt,radius,'symmetric');
    gt_im = I;
    ps_bd=zeros(k1,patchSiz^2,'single');
    ps_rgb=zeros(k1,3*patchSiz^2,'single');
    
    p=patchSiz-1;
    for j=1:k1
        ps(:,:,:,j)=chns(xy(j,2):xy(j,2)+p,xy(j,1):xy(j,1)+p,:);
        ps_bd(j,:) = reshape(gt_m(xy(j,2):xy(j,2)+p,xy(j,1):xy(j,1)+p),1,[]);
        ps_rgb(j,:) = reshape(gt_im(xy(j,2):xy(j,2)+p,xy(j,1):xy(j,1)+p,:),1,[]);        
    end
    
    if(0), montage2(squeeze(ps(:,:,1,:))); drawnow; end
    
    % compute features and store
    ftrs1=[reshape(ps,[],k1)' U_stComputeSimFtrs(ps,opts)];
    ftrs(k+1:k+k1,:) = ftrs1(:,fids);
    ftrs_bd(k+1:k+k1,:) = ps_bd;
    ftrs_rgb(k+1:k+k1,:) = ps_rgb;
    labels(k+1:k+k1) = lbls;
    k=k+k1;
    tocStatus(tid,i/nImgs);
end
if k<size(ftrs,1)
    ftrs=ftrs(1:k,:);
    ftrs_bd=ftrs_bd(1:k,:);
    labels=labels(1:k);
end


%{
train_id = zeros(1,31000);
test_id = zeros(1,7000);
for i=1:150
    ind = find(labels == i);
    train_id((i-1)*100+(1:100)) = ind(1:100);
    test_id((i-1)*20+(1:20)) = ind(101:120);   
end
    ind = find(labels == 151);
    train_id(150*100+(1:80*200)) = ind(1:80*200);
    test_id(150*20+(1:20*200)) = ind(80*200+(1:20*200));

intersect(train_id,test_id)
numel(unique(train_id))+numel(unique(test_id))

% for python
train_im = single([labels(train_id)-1 ftrs(train_id,1:3*35^2)])';
save train_im train_im
test_im = single([labels(test_id)-1 ftrs(test_id,1:3*35^2)]);
save test_im test_im
train_bd = single([ftrs_bd(train_id,:)]);
save train_bd train_bd
test_bd = single([ftrs_bd(test_id,:)])';
save test_bd test_bd

% for matlab
fid = fopen('st_train.bin', 'w');
fwrite(fid,[labels(train_id)-1 ftrs(train_id,:)], 'single');
fclose(fid);
fid = fopen('st_test.bin', 'w');
fwrite(fid,[labels(test_id)-1 ftrs(test_id,:)], 'single');
fclose(fid);
%}
switch did
case -1
    % train
    train_im = single([labels-1 ftrs_rgb])';
    %p_save(train_im,1,10,'train_im_bp');    
    save(['train_im_bn' num2str(nid)],'train_im')
    %train_feat = single([labels-1 ftrs])';
    %save -v7.3 train_feat train_feat
    train_bd = single([ftrs_bd])';
    %p_save(train_b,1,10,'train_bd_bp');    
    save(['train_bd_bn' num2str(nid)],'train_bd')
    
    % for matlab
    fid = fopen(['st_train_bp.bin' num2str(nid)], 'w');
    fwrite(fid,[labels-1 ftrs], 'single');
    fclose(fid);
    case 0
    % train
    train_im = single([labels-1 ftrs_rgb])';
    %p_save(train_im,1,10,'train_im_bp');    
    save train_im_bp train_im
    %train_feat = single([labels-1 ftrs])';
    %save -v7.3 train_feat train_feat
    train_bd = single([ftrs_bd])';
    %p_save(train_b,1,10,'train_bd_bp');
    save train_bd_bp train_bd
    
    % for matlab
    fid = fopen('st_train_bp.bin', 'w');
    fwrite(fid,[labels-1 ftrs], 'single');
    fclose(fid);
case 1
    % train
    train_im = single([labels-1 ftrs_rgb])';
    save train_im train_im
    %train_feat = single([labels-1 ftrs])';
    %save -v7.3 train_feat train_feat
    train_bd = single([ftrs_bd])';
    save train_bd train_bd
    % for matlab
    fid = fopen('st_train.bin', 'w');
    fwrite(fid,[labels-1 ftrs], 'single');
    fclose(fid);
case 2
    % valid
    test_im = single([labels-1 ftrs_rgb]);
    save test_im test_im
    test_bd = single([ftrs_bd])';
    save test_bd test_bd
    %test_feat = single([labels-1 ftrs])';
    %save -v7.3 test_feat test_feat
    % for matlab
    fid = fopen('st_test.bin', 'w');
    fwrite(fid,[labels-1 ftrs], 'single');
    fclose(fid);
end



%{
% 31,000 * (35^2*3)+1
fid = fopen('occ_train.bin', 'w');
fwrite(fid,[labels-1 ftrs(:,1:35^2)], 'single');
fclose(fid);

fid = fopen('occ_train_bd.bin', 'w');
fwrite(fid,[labels-1 ftrs_bd], 'single');
fclose(fid);

% test 
num=10000;
fid = fopen('occ_train2.bin', 'w');
fwrite(fid,[labels(1:num)-1 ftrs(1:num,:)], 'single');
fclose(fid);


% check data
load data/train/train_im
load data/train/train_bd
id=10;
subplot(211),imagesc(reshape(train_bd(:,id),[35 35]))
subplot(212),imagesc(uint8(reshape(train_im(2:end,id),[35 35 3])))

% rgb data
ndim=psz^2;
ntrain=size(train_im,2);
train_imc = reshape(permute(reshape(train_im(2:end,:),[ndim,3 ntrain]),[2 1 3]),3*ndim,[]);
ppsz =[psz,psz];
imagesc(uint8(cat(3,reshape(train_imc(1:3:end,1),ppsz),reshape(train_imc(2:3:end,1),ppsz),reshape(train_imc(3:3:end,1),ppsz))))

%pca
ndim = 3000;
% scale to 0-1
fid = fopen('data/test/st_test.bin');
data = fread(fid, [7000 inf], 'single');
fclose(fid);
lb = data(:,1);
data = data(:,2:end);
min_d = min(data);
max_d = max(data);
data = bsxfun(@minus,data,min_d);
data = bsxfun(@rdivide,data,max_d);
tmp_cov = cov(data);
[aa,bb]= eigs(tmp_cov,ndim);

test_feat_s = [lb data*aa];
save test_feat_s test_feat_s
save eigen_feat aa bb

fid = fopen('data/train/st_train.bin');
data = fread(fid, [31000 inf], 'single');
fclose(fid);
lb = data(:,1);
data = data(:,2:end);
data = bsxfun(@minus,data,min_d);
data = bsxfun(@rdivide,data,max_d);
train_feat_s = [lb data*aa]; 
save train_feat_s train_feat_s


id=0;U_data;
id=-1;for nid=1:8;U_data;end


%}
