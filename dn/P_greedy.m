addpath('../util')
init
addpath([VLIB 'Opt/Axb'])
addpath(genpath([VLIB 'Low/Denoise']))
if ~exist('ll','var')
    ll = -1;
end
if ~exist('pss','var')
    load([VOC_DIR 'c_voc_p1'])
end
if ~exist('npss','var')
    load([VOC_DIR 'n_voc_p1'])
end
s = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(s);
sz = size(pss);
p_eval=1;

num_in = size(npss,1);
num_out = size(pss,1);
num_train = 1e5;%size(pss,2);
num_test = 1e5;%size(pss,2);
if ~exist('mat_x','var')
mat_x = (double(npss(:,1:num_train))/255-0.5)/0.2;
mat_y = (double(pss(:,1:num_train)')/255-0.5)/0.2;
mat_xx = (double(npss(:,num_train+(1:num_test)))/255-0.5)/0.2;
mat_yy = (double(pss(:,num_train+(1:num_test))')/255-0.5)/0.2;
end

% init
if(ll==1 || ll==0)
    if ~exist(['init_p' num2str(ll) '.mat'],'file')
        num_h = 1000;
        init_id = 0;
        switch init_id
        case 0
            std_W0 = 0.01; mu_W0 = 0;
            W0 = single(rand(num_in,num_h)*std_W0+mu_W0);
            std_b0 = 0.01; mu_b0 = 0;
            b0 = single(rand(1,num_h)*std_b0+mu_b0);
        case 1
            W0 = zeros(num_in,num_h,'single');
            W0(1:num_in+1:end) = 0.2;
            b0 = zeros(1,num_h,'single');
        case 2
            std_W0 = 0.2; mu_W0 = 0.5;
            W0 = single(randn(num_in,num_h)*std_W0+mu_W0);
            std_b0 = 0.01; mu_b0 = 0;
            b0 = single(randn(1,num_h)*std_b0+mu_b0);
        end
        pp = bsxfun(@plus,mat_x*W0,b0);
        if ll==0
            mat_x2 = sigmf(pp,[1 0]);
        else
            mat_x2 = tanh(pp);
        end
        tmp_one = ones(num_train,1,'single');
        init_id2 = 1;
        param = cell(2);
        param(1,:) = { single(W0), single(b0)};
        switch init_id2
        case 0
            rr=rank([mat_x,tmp_one]'*[mat_x,tmp_one])
            if rr<num_h
                error('rank not full')
            end
            x0 = zeros(1+num_h,num_in,'single');
            x0(1:(num_h+2):end) = 1;
            opt = 'backslash';
            xx= U_axb([mat_x,tmp_one]'*[mat_x,tmp_one], [mat_x,tmp_one]'*mat_y,x0,opt);
            param(2,:) = { single(xx(1:end-1,:)), single(xx(end,:))};
        case 1
            xx = ([mat_x2(:,1:num_out), tmp_one])\mat_y;
            param(2,:) = { [single(xx(1:end-1,:));zeros(num_h-num_out,num_out,'single')], single(xx(end,:))};
        end
        save(['init_p' num2str(ll)],'param');
    end
    if p_eval
        U_eval(ll,mat_x,mat_y);
    end
end
num_l = size(param,1);
% back prop
% naive pass back 0th order
    tmp_y = mat_y;
   for j = num_l:-1:2
       tmp_yy = param{j,1}'\(bsxfun(@minus,tmp_y,param{j,2}))';
       rescale = max(abs(tmp_yy(:)))*1.1;
       param{j,1} = param{j,1}*rescale;
       tmp_y = atanh(tmp_yy/rescale)';
       %{
       tmp_b = double(bsxfun(@minus,tmp_y,param{j,2}))';
       tmp_yy = lsqlin(double(param{j,1}'),tmp_b(:,1),[],[],[],[],-1,1);
       tmp_yy = lsqlin(double(param{j,1}'),double(bsxfun(@minus,tmp_y,param{j,2}))',[],[],-1,1);
       %}
       tmp_p = [mat_x,ones(num_train,1,'single')]\tmp_y;
       param{j-1,1} = tmp_p(1:end-1,:);
       param{j-1,2} = tmp_p(end,:);
   end 
U_eval(param,mat_x,mat_y)
U_eval(param,mat_xx,mat_yy)
