init
addpath([VLIB 'Opt/Axb'])
addpath(genpath([VLIB 'Low/Denoise']))

if ~exist('init_p0.mat','file')
    if ~exist('pss','var')
        load([VOC_DIR 'c_voc_p1'])
    end
    if ~exist('npss','var')
        load([VOC_DIR 'n_voc_p1'])
    end
    sz = size(pss);
    A = [double(npss)/255;ones(1,sz(2))]*[double(npss)/255;ones(1,sz(2))]';
    b = [double(npss)/255;ones(1,sz(2))]*double(pss)'/255;
    x0 = zeros(sz(1)+1,sz(1));
    x0(1:(sz(1)+2):end) = 1;
    opt = 'pcg';
    opt = 'backslash';
    xx= U_axb(A,b,x0,opt);
    param ={ single(xx(1:end-1,:)), single(xx(end,:))};
    save init_p0 param
    % initial  error: 0.89 per patch
    mean(sum(((single(npss) - single(pss))/255).^2,1))
    % training error: 0.1915 per patch
    xx = [param{1};param{2}];
    yhat = [single(npss)/255;ones(1,size(npss,2))]' * xx;
    err1 = abs((yhat - single(pss)'/255)); 
    sum(sum(err1(1:10,:))
    err = ((yhat - single(pss)'/255)).^2; 
    sum(sum(err(1:10,:))

    mean(sum(err,2))
    mean(sum(err(1:10,:),2))

    %{
    yhat(1:10,1)
    pp=sum(reshape(sum((single(pss)/255).^2),10,[]))
    %}
end
