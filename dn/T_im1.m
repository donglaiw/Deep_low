init
% extract commonly used options
psz=17;
sig = 25;
id=1e4+10;
if ~exist([T_DIR 'test_p' num2str(id) '.mat'],'file')
    imgIds=dir([DATA_VOC '*.jpg']);
    imgIds={imgIds.name};
    I = rgb2gray(imread([DATA_VOC imgIds{id}]));
    sz = size(I);
    ps = im2col(I,[psz psz]);
    In = uint8(single(I)+single(randn(sz)*sig));
    nps = im2col(In,[psz psz]);
    save([T_DIR 'test_p' num2str(id)],'ps','nps','sz');
    save([T_DIR 'test_i' num2str(id)],'I','In');
    imwrite(uint8(I),[T_DIR 'I_' num2str(id) '.png'])
    imwrite(uint8(In),[T_DIR 'In' num2str(id) '.png'])
end



id = 10010;
psz = 17;
sig = 25;
if ~exist('I','var')
    load([T_DIR 'test_i' num2str(id)]);
    sz = size(I);
end
    
%err= zeros(1,5);
%yhats= cell(1,5);
for algo = [3]%0:5
    yhat = In;
    switch algo    
    case 1
        % optimal L2
        if ~exist('ps','var')
            load([T_DIR 'test_p' num2str(id)]);
        end
        % initial test image error: 0.9269
        %mean(sum(((single(nps) - single(ps))/255).^2,1))
        load init_p-1
        phat = [double(nps);ones(1,size(nps,2))]'*[param{1};param{2}];
        yhat = uint8(scol2im(phat',psz,sz(1),sz(2),'average'));
        % initial test image error: 0.1715
        %mean(sum((single(phat)/255-single(ps)'/255).^2,2))
    case 2
         [~, yhat] = BM3D(1, double(In)/255, sig, 'np',0);
         yhat = uint8(yhat*255);
    case 3
        sz = size(I);
        exp_id = 0; 
        model_id = 1;
        num_epoch = 10000;
        nn = sprintf('dl_r%d_%d_%d.mat',exp_id,model_id,num_epoch);
        if exist(nn,'file')
            load(nn) 
            yhat=scol2im(result{1}',psz,sz(1),sz(2),'average');
        end
        
    case 4
        exp_id = 0; 
        model_id = 1;
        num_epoch = 100;
        nn = sprintf('dl_r%d_%d_%d.mat',exp_id,model_id,num_epoch);
        if exist(nn,'file')
            load(nn) 
            yhat=scol2im(result{1}',psz,sz(1),sz(2),'average');
        end        
    case 5
        addpath([VLIB 'DeepL/dn_mlp'])
        % define some parameters for denoising
        model = {};
        % width of the Gaussian window for weighting output pixels
        model.weightsSig = 2;
        % the denoising stride. Smaller is better, but is computationally 
        % more expensive.
        model.step = 1;
        % denoise
        fprintf('Starting to denoise...\n');
        yhat = fdenoiseNeural(single(In), 25, model);
    end
    err(algo) = U_psnr(I,yhat);
    yhats{algo} = yhat;
end
%{
for algo=1:5
    imwrite(yhats{algo},[T_DIR 'Ir_' num2str(algo) '.png'])
end
%}
