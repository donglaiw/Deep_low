init
addpath([VLIB 'Opt/Axb'])
addpath(genpath([VLIB 'Low/Denoise']))

if ~exist('l2opt.mat','file')
    if ~exist('pss','var')
        load([VOC_DIR 'c_voc_p1'])
    end
    if ~exist('npss','var')
        load([VOC_DIR 'n_voc_p1'])
    end
    sz = size(pss);
    A = [double(npss);ones(1,sz(2))]*[double(npss);ones(1,sz(2))]';
    b = [double(npss);ones(1,sz(2))]*double(pss)';
    x0 = zeros(sz(1)+1,sz(1));
    x0(1:(sz(1)+2):end) = 1;
    opt = 'pcg';
    opt = 'backslash';
    xx= U_axb(A,b,x0,opt);
    save l2opt xx
    % initial  error: 0.89 per patch
    mean(sum(((single(npss) - single(pss))/255).^2,1))
    % training error: 0.1915 per patch
    mean(sum((([single(npss);ones(1,size(npss,2))]' * xx - single(pss)')/255).^2,2))
else 
    load l2opt
end

id = 10010;
psz = 17;
sig = 25;
if ~exist('I','var')
    load([T_DIR 'test_i' num2str(id)]);
end
err= zeros(1,5);
yhats= cell(1,5);
for algo =0:5
    switch algo
    case 0 
        yhat = In;
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
    end
    err(algo) = U_psnr(I,yhat);
    yhats{algo} = algo;
end

for algo=1:5
    imwrite(yhats{algo},[T_DIR 'Ir_' num2str(algo) '.png'])
end
