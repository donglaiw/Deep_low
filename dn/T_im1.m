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
    In = uint8(single(I)+single(rand(sz)*sig));
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
    
err= zeros(1,5);
yhats= cell(1,5);
for algo =0:5
    yhat = In;
    switch algo    
    case 1
        % optimal L2
        if ~exist('ps','var')
            load([T_DIR 'test_p' num2str(id)]);
        end
        % initial test image error: 0.9269
        %mean(sum(((single(nps) - single(ps))/255).^2,1))
        phat = [double(nps);ones(1,size(nps,2))]'*xx;
        yhat = uint8(scol2im(phat',psz,sz(1),sz(2),'average'));
        % initial test image error: 0.1715
        %mean(sum((single(phat)/255-single(ps)'/255).^2,2))
    case 2
         [~, yhat] = BM3D(1, double(In)/255, sig, 'np',0);
         yhat = uint8(yhat*255);
    case 3
        exp_id = 0; 
        model_id = -1;
        num_epoch =1000;
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
    end
    err(algo) = U_psnr(I,yhat);
    yhats{algo} = algo;
end

for algo=1:5
    imwrite(yhats{algo},[T_DIR 'Ir_' num2str(algo) '.png'])
end

    % theano prediction
    %mean(sum(abs(result{1}'-single(ps/255))))
    U_psnr(I,c)
    imwrite(uint8(255*c),sprintf([T_DIR 'Ir_%d_%d_%d_%d.png'],id,exp_id,model_id,num_epoch))
end
