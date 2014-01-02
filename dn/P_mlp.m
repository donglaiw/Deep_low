addpath('../')
init
addpath([VLIB 'Opt/Axb'])
addpath(genpath([VLIB 'Low/Denoise']))
ll = -2;
if ~exist('pss','var')
    load([VOC_DIR 'c_voc_p1'])
end
if ~exist('npss','var')
    load([VOC_DIR 'n_voc_p1'])
end
s = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(s);
sz = size(pss);
peval=1;

num_in = size(npss,1);
num_out = size(pss,1);
num_train = size(pss,2);
mat_x = (double(npss)/255-0.5)/0.2;
mat_y = (double(pss')/255-0.5)/0.2;

if ll==-2
      load([VLIB 'DeepL/dn_mlp/weights_cvpr.mat']);
      part = (double(npss)/255-0.5)/0.2;
      for i=1:length(w)
      part = [part;ones(1,size(npss,2))];
      if (i<length(w))
        part = tanh(w{i}*part);
      else
        p_final = w{i}*part;
      end
    end
        err = ((p_final*0.2+0.5 - double(pss)/255)).^2; 
        % 0.5926
        mean(sum(err,1))

elseif ll==-1
    if ~exist('init_p-1.mat','file')
        %{
        A = [double(npss)/255;ones(1,sz(2))]*[double(npss)/255;ones(1,sz(2))]';
        b = [double(npss)/255;ones(1,sz(2))]*double(pss)'/255;
        x0 = zeros(sz(1)+1,sz(1));
        x0(1:(sz(1)+2):end) = 1;
        opt = 'backslash';
        xx= U_axb(A,b,x0,opt);
        %}
        xx = [mat_x;ones(1,size(npss,2))]'\mat_y;
        param ={ single(xx(1:end-1,:)), single(xx(end,:))};
        save init_p-1 param
        %{
        % initial  error: 2.53 per patch
        mean(sum(((single(npss) - single(pss))/255).^2,1))
        % training error: 0.9066 per patch
        load init_p-1
        xx = [param{1};param{2}];
        yhat = [mat_x;ones(1,size(npss,2))]' * xx;
        err = (yhat - mat_y).^2; 
        mean(sum(err,2))
        mean(sum(err,2))
        mean(sum(err(1:10,:),2))
        yhat(1:10,1)
        pp=sum(reshape(sum((single(pss)/255).^2),10,[]))
        %}
    end
elseif(ll==1 || ll==0)
    if ~exist(['init_p' num2str(ll) '.mat'],'file')
        num_h = 1000;
        init_id = 1;
        switch init_id
        case 0
            std_W0 = 0.006; mu_W0 = 0;
            W0 = single(rand(num_in,num_h)*std_W0+mu_W0);
            std_b0 = 0.8; mu_b0 = -0.4;
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
        pp = bsxfun(@plus,mat_x'*W0,b0);
        if ll==0
            mat_x = sigmf(pp,[1 0]);
        else
            mat_x = tanh(pp);
        end
        tmp_one = ones(num_train,1,'single');
        init_id2 = 1;
        param = cell(2);
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
            param(1,:) = { single(W0), single(b0)};
            param(2,:) = { single(xx(1:end-1,:)), single(xx(end,:))};
        case 1
            xx = ([mat_x(:,1:289), tmp_one])\mat_y;
            param(1,:) = { single(W0), single(b0)};
            param(2,:) = { [single(xx(1:end-1,:));zeros(711,289,'single')], single(xx(end,:))};
        end
        save(['init_p' num2str(ll)],'param');
        if peval
            %yhat = bsxfun(@plus,pp*param{2,1},param{2,2});
            %yhat = bsxfun(@plus,bsxfun(@plus,mat_x*param{1,1},param{1,2}) * param{2,1},param{2,2});
            %yhat = bsxfun(@plus,bsxfun(@plus,mat_x*param{1,1},param{1,2}) * param{2,1},param{2,2});
            yhat = [mat_x(:,1:289),tmp_one] * xx;
            err = sum((yhat - single(pss)'/255).^2,2); 
        end
    end
elseif (ll==2 || ll==3)
        num_h = [num_in,500,500,num_out];
        W=cell(1,2);
        b=cell(1,2);
        mat_x = mat_x';
        for i=1:2
            W{i} = zeros(num_h(i),num_h(i+1),'single');
            W{i}(1:num_h(i)+1:end) = 0.2;
            b{i} = zeros(1,num_h(i+1),'single');
            mat_x = bsxfun(@plus,mat_x*W{i},b{i});
            if ll==2
                mat_x = sigmf(mat_x,[1 0]);
            else
                mat_x = tanh(mat_x);
            end
        end

        tmp_one = ones(num_train,1,'single');
        param = cell(2);
        xx = ([mat_x(:,1:num_h(end)), tmp_one])\mat_y;
        param(1,:) = { single(W{1}), single(b{1})};
        param(2,:) = { single(W{2}), single(b{2})};
        param(3,:) = { [single(xx(1:end-1,:));zeros(num_h(3)-num_h(end),num_h(end),'single')], single(xx(end,:))};
        save(['init_p' num2str(ll)],'param');
        peval=0;
        if peval
            %yhat = bsxfun(@plus,pp*param{2,1},param{2,2});
            %yhat = bsxfun(@plus,bsxfun(@plus,mat_x*param{1,1},param{1,2}) * param{2,1},param{2,2});
            yhat = [mat_x(:,1:num_h(end)),tmp_one] * xx;
            err = sum((yhat - single(pss)'/255).^2,2); 
        end
    end

