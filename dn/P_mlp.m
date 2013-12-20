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
end
