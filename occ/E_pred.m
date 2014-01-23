addpath('../util');init
% benchmark for comparison of patch label prediction
if ~exist('did','var')
did = 3;
end
if ~exist('algo','var')
algo = 1;
end
if ~exist('data','var')
    switch did
    case 1
        fid = fopen('data/test/st_test.bin');
        data = fread(fid, [7000 inf], 'single');
        fclose(fid);
    case 2
        fid = fopen('data/train/st_train.bin');
        data = fread(fid, [31000 inf], 'single');
        data = data(1:10:end,:);
        fclose(fid);
    case 3
        fid = fopen('data/train/st_train.bin');
        data = fread(fid, [31000 inf], 'single');        
        data = data(setdiff(1:size(data,1),1:10:size(data,1)),:);
        fclose(fid);
    case 4
        if ~exist('train_im','var')
            load data/train/train_im
        end
        data = train_im(2:end,1:10:end)';
    case 5
        if ~exist('train_im','var')
            load data/train/train_im
        end
        data = train_im(2:end,setdiff(1:size(train_im,2),1:10:size(train_im,2)))';
    end
end

    num_test = size(data,1);

switch algo
case 1
    if ~exist('model','var')
        % small: .6511/0.7037(test) 
        load([VLIB 'Mid/Boundary/SketchTokens/models/forest/modelSmall.mat']);
        %model = U_st1(model);
        % big: .7428/.7766(test) 
        %load([VLIB 'Mid/Boundary/SketchTokens/models/forest/modelFull.mat']);
    end
    addpath(genpath([VLIB 'Mid/Boundary/SketchTokens/']));
    addpath(genpath([VLIB '../Piotr/']));

    % load all feature    
    if did<=3
        im = single(permute(reshape(data(:,1+(1:17150))', 35, 35, 14,[]),[1 2 4 3]));
        st = U_stDetect( reshape(im,35,35*num_test,14), model, 35 );
    else
        % lost bounday info for filtering
        im = uint8(permute(reshape(data', 35, 35, 3,[]),[1 2 4 3]));
        st = U_stDetect( reshape(im,35,35*num_test,3), model, 35 );
    end
    st = squeeze(st);
    [~,ll] = max(st,[],2);
    tt= data(:,1)+1;
    % 151 classes
    1-sum(ll-tt==0)/num_test
    % 2 classes
    st2 =[sum(st(:,1:end-1),2) st(:,end)];
    [~,ll2] = max(st2,[],2);
    tt= data(:,1)+1;
    tt(tt~=151)=1;
    tt(tt==151)=2;
    1-sum(ll2-tt==0)/num_test
end
