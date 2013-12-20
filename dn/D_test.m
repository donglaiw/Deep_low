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

exp_id = 0; 
model_id = -1;
num_epoch =1000;
nn = sprintf('dl_r%d_%d_%d.mat',exp_id,model_id,num_epoch);
if exist(nn,'file')
    load(nn) 
    if ~exist('I','var')
        load([T_DIR 'test_i' num2str(id)],'I');
        sz = size(I);
    end
    c=scol2im(result{1}',psz,sz(1),sz(2),'average');
    % theano prediction
    %mean(sum(abs(result{1}'-single(ps/255))))
    U_psnr(I,c)
    imwrite(uint8(255*c),sprintf([T_DIR 'Ir_%d_%d_%d_%d.png'],id,exp_id,model_id,num_epoch))
end
