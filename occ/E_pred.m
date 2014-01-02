addpath('../')
init
% benchmark for comparison of patch label prediction
did = 1;
algo = 1;
if ~exist('data','var')
    switch did
    case 1
        fid = fopen('data/test/st_test.bin');
        data = fread(fid, [7000 inf], 'single');
        fclose(fid);
    case 2
        fid = fopen('data/train/st_train.bin');
        data = fread(fid, [31000 inf], 'single');
        fclose(fid);
    end
    num_test = size(data,1);
end


switch algo
case 1
    if ~exist('model','var')
        % small: .6511/0.7037(test) 
        %load([VLIB 'Mid/Boundary/SketchTokens/models/forest/modelSmall.mat']);
        % big: .7428/.7766(test) 
        load([VLIB 'Mid/Boundary/SketchTokens/models/forest/modelFull.mat']);
    end
    addpath(genpath([VLIB 'Mid/Boundary/SketchTokens/']));
    addpath(genpath([VLIB '../Piotr/']));

    % load all feature    
    im = single(permute(reshape(data(:,1+(1:17150))', 35, 35, 14,[]),[1 2 4 3]));
    st = U_stDetect( reshape(im,35,35*num_test,14), model, 35 );
    [~,ll] = max(squeeze(st),[],2);
    % 151 classes
    % sum(ll-data(test_id,1)-1==0)/num_test
    % 2 classes
    tt= data(:,1)+1;
    tt(tt~=151)=0;
    ll2=ll; ll2(ll~=151)=0;
    sum(ll2-tt==0)/num_test
end
