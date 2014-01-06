addpath('../util');init
if ~exist('ll','var')
    ll = 3;
end
if ~exist('train_im','var')
    load([TR_DIR 'train_im'])
end
if ~exist('train_bd','var')
    load([TR_DIR 'train_bd'])
end
s = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(s);
sz = size(train_im);
p_eval=1;
num_in = size(train_im,1)-1;
num=31e3;%1e4;%
ind_test = 1:10:num;%size(pss,2);
ind_train = setdiff(1:num,ind_test);%size(pss,2);
num_train = numel(ind_train);
num_test = numel(ind_test);
mat_x = (single(train_im(2:end,ind_train))/255-0.5)/0.2;
mat_y = single(train_bd(:,ind_train)');
mat_xx = (single(train_im(2:end,ind_test))/255-0.5)/0.2;
mat_yy = single(train_bd(:,ind_test)');
crop = 1;
if crop
    x =4;y=4;
    sz1=35;sz2=28;
    crop_id = bsxfun(@plus,x+(1:sz2)-1,(y+(1:sz2)'-1-1)*sz1)'; 
    crop_id = crop_id(:);
    mat_y = mat_y(:,crop_id);
    mat_yy = mat_yy(:,crop_id);
end
num_out = size(mat_y,1);
if ll==-1
    if ~exist('init_p-1.mat','file')
        xx = [mat_x;ones(1,size(mat_x,2),'single')]'\mat_y;
        param ={ single(xx(1:end-1,:)), single(xx(end,:))};
        save init_p-1 param
        %{
        % baseline: 
        % all 0: (sparse) 11.4722
        mean(sum(mat_y.^2,2))
        %}
    end
    if p_eval
        load init_p-1
        xx = [param{1};param{2}];
        % training error: 9.4196
        yhat = [mat_x;ones(1,size(mat_x,2))]' * xx;
        err = (yhat - mat_y).^2; 
        mean(sum(err,2))

        yhat2 = [mat_xx;ones(1,size(mat_xx,2))]' * xx;
        err2 = (yhat2 - mat_yy).^2; 
        % valid error: 13.3616
        mean(sum(err2,2))
    end 
elseif(ll==3)
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
        mat_x2 = tanh(pp);
        tmp_one = ones(num_train,1,'single');
        param = cell(2);
        param(1,:) = { single(W0), single(b0)};
        xx = ([mat_x2(:,1:min(num_h,num_out)), tmp_one])\mat_y;
        param(2,:) = { [single(xx(1:end-1,:));zeros(num_h-num_out,num_out,'single')], single(xx(end,:))};
        save(['init_p' num2str(ll)],'param');
    end
    if p_eval
        load(['init_p' num2str(ll)],'param');
        pp = bsxfun(@plus,mat_x'*param{1,1},param{1,2});
        pp2 = bsxfun(@plus,mat_xx'*param{1,1},param{1,2});
        mat_x2 = tanh(pp);
        mat_xx2 = tanh(pp2);
        yhat = [mat_x2,ones(size(mat_x2,1),1,'single')] * [param{2,1}; param{2,2}];
        yhat2 = [mat_xx2,ones(size(mat_xx2,1),1,'single')] * [param{2,1}; param{2,2}];
        % training error: 10.7768 per patch
        err = (yhat - mat_y).^2; 
        mean(sum(err,2))
        % valid error: 11.8577 per patch
        err2 = (yhat2 - mat_yy).^2; 
        mean(sum(err2,2))

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
        p_eval=1;
        if p_eval
            % 0.2065
            %yhat = bsxfun(@plus,pp*param{2,1},param{2,2});
            %yhat = bsxfun(@plus,bsxfun(@plus,mat_x*param{1,1},param{1,2}) * param{2,1},param{2,2});
            yhat = [mat_x(:,1:num_h(end)),tmp_one] * xx;
            err = sum(((yhat*0.2+0.5) - single(pss)'/255).^2,2); 
        end
    end

