addpath('../')
init
% benchmark for comparison of patch label prediction
did=1;
algo = 1;
if ~exist('data','var')
    switch did
    case 1
        fid = fopen('data/st_test.bin');
        data = fread(fid, [7000 inf], 'single');
        fclose(fid);
    end
end


switch algo
case 1
    if ~exist('model','var')
        % small: 51.32
        load([VLIB 'Mid/Boundary/SketchTokens/models/forest/modelSmall.mat']);
        % big: 51
        %load([VLIB 'Mid/Boundary/SketchTokens/models/forest/modelfull.mat']);
    end
    addpath(genpath([VLIB 'Mid/Boundary/SketchTokens/']));
    addpath(genpath([VLIB 'Piotr/']));

    % load all feature    
    im = single(permute(reshape(data(:,1+(1:17150))', 35, 35, 14,[]),[1 2 4 3]));
    st = U_stDetect( reshape(im,35,35*numel(test_id),14), model, 35 );
    [~,ll] = max(squeeze(st),[],2);
    % 151 classes
    % sum(ll-data(test_id,1)-1==0)/numel(test_id)
    % 2 classes
    tt= data(:,1)+1;
    tt(tt~=151)=0;
    ll2=ll; ll2(ll~=151)=0;
    sum(ll2-tt==0)/numel(test_id)
end
