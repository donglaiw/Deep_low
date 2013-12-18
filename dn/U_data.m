

stream=RandStream('mrg32k3a','Seed',1);
set(stream,'Substream',1);
RandStream.setGlobalStream( stream );

DATA_UCB = '/home/Stephen/Desktop/Data/Seg/BSR/BSDS500/data/';
trnImgDir = [DATA_UCB '/images/train/'];
imgIds=[dir([trnImgDir '*.jpg']) dir([trnImgDir '*.jpg'])];

imgIds={imgIds.name};
nImgs=length(imgIds);

% extract commonly used options
patchSiz=17;
ps =cell(1,nImgs);
ff=1;
step = 3;
parfor i = 1:nImgs
    I = imread([trnImgDir imgIds{i}]);    
    switch ff
        case 0
            ps{i} = [im2col(I(:,:,1),[patchSiz patchSiz],'distinct');...
                im2col(I(:,:,2),[patchSiz patchSiz],'distinct');...
                im2col(I(:,:,3),[patchSiz patchSiz],'distinct')];
        case 1
            tmp = [im2col(I(:,:,1),[patchSiz patchSiz]);...
                im2col(I(:,:,2),[patchSiz patchSiz]);...
                im2col(I(:,:,3),[patchSiz patchSiz])];
            sz = size(I);
            tmp = reshape(tmp,[patchSiz^2*3,sz(1:2)-patchSiz+1]);
            ps{i} = reshape(tmp(:,1:step:end,1:step:end),patchSiz^2*3,[]);
    end
    %imagesc(reshape(ps(:,100),[17 17 3]))
end
sum(arrayfun(@(x) size(ps{x},2),1:nImgs))

pss = cell2mat(ps);
save -v7.3 dn_ucb_ pss;

% 31,000 * 35^2+1
fid = fopen(['occ_train' num2str(opts.nNeg) '.bin'], 'w');
fwrite(fid,[labels-1 ftrs(:,1:35^2)], 'single');
fclose(fid);
fid = fopen(['occ_train_bd' num2str(opts.nNeg) '.bin'], 'w');
fwrite(fid,[labels-1 ftrs_bd], 'single');
fclose(fid);

%{
num=15000;
fid = fopen('occ_train2.bin', 'w');
fwrite(fid,[labels(1:num)-1 ftrs(1:num,:)], 'single');
fclose(fid);
%}