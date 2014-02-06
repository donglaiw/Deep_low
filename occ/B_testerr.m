% train_id=10;test_id=2;im_id=1;mlp_id=3;dim_n=2;T_mlp;imagesc(pb0)
addpath('../util');init
addpath(genpath([VLIB '../Piotr']))
addpath([VLIB 'Mid/Boundary/SketchTokens'])
% extract commonly used options



psz = 35;
algo_id = 1;
for test_id=2:2
    for mlp_id=-1:3
        if mlp_id<=0
            switch mlp_id
            case -1
                load('st_bd')
            case 0
                load('st_bd_2')
            end
            err = zeros(1,10);
            for im_id = 1:10
                sz = size(Is{im_id});
                gt_b=sum(cat(3,gts{im_id}{:}),3);
                gt_id = gt_b~=0;
                pre_y = (sts{im_id}==151)+1;
                err(im_id) = (nnz(pre_y(gt_id)==2)+nnz(pre_y==1)-nnz(pre_y(gt_id)==1))/ prod(sz(1:2)); 
            end
            %err
            [mlp_id mean(err)]
        else
            for train_id=9:10
                switch train_id
                    case 9
                        dim_n=151;
                    case 10
                        dim_n=2;
                end
                switch mlp_id
                    case 1
                        model_id = 0;
                        num_dim = [500 dim_n];
                        num_epoch = 100;
                    case 2
                        model_id = 1;
                        num_dim = [500 500 dim_n];
                        num_epoch = 100;
                    case 3
                        model_id = 2;
                        num_dim = [500 500 500 dim_n];
                        num_epoch = 100;
                end

        err = zeros(1,10);
        for im_id = 1:10
            num = numel(Is);
            pb = cell(1,num);
            str_dim = num2str(num_dim,'%d_');
            str_dim(str_dim==' ')=[];
            nn = [sprintf('result/%d_%d_',algo_id,model_id) str_dim sprintf('%d_%d/dl_r%d_',train_id,num_epoch,test_id)];
            sz = size(Is{im_id});
            if exist([nn num2str(im_id-1) '.mat'],'file')
                %python P_occ.py 1 2 100 500,500,500,151 9 2
                tmp=load([nn num2str(im_id-1)]); 
                if size(tmp.result{1},2)>2
                    [~,pred_y] = max([sum(tmp.result{1}(:,1:end-1),2) tmp.result{1}(:,end)],[],2);
                else
                    [~,pred_y] = max(tmp.result{1},[],2);
                end
                pre_y = reshape(pred_y,sz([2 1]))';
                pb0 = stToEdges( reshape(tmp.result{1},[sz([2 1]) num_dim(end)]), 1 )';
            else
                disp(['no exist'])
            end
            err(im_id) = (nnz(pre_y(gt_id)==2)+nnz(pre_y==1)-nnz(pre_y(gt_id)==1))/ prod(sz(1:2));
        end
        %err
        [train_id mlp_id mean(err)]
    end
    end
    end
    end
