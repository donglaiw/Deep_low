
tid = 2;
did = 0;

load jo;
switch tid 
    case 1 
        %mat_x = rand(3,1);mat_y=rand();save db mat_x mat_y
        yhat = conv(mat_x,result{1}(end:-1:1),'valid')*result{2,1};
        (yhat-mat_y)^2
    case 2
        %mat_x = rand(12,1);mat_y=rand();save db_2 mat_x mat_y
        %yhat = conv(mat_x,reshape(result{1},1,[]),'valid')*result{2,1};
        %yhat = conv(mat_x,reshape(result{1}(end:-1:1,:,:),1,[]),'valid')*result{2,1};
        %yhat = conv(mat_x,reshape(permute(result{1}(end:-1:1,:,:),[2 3 1]),1,[]),'valid')*result{2,1};
        switch did
        case 2
            tmp = load('db_11');
            tmp_x = tmp.mat_x;
            tmp_y = tmp.mat_y;
        case 0
            tmp_x = mat_x(:,1:10:2000);
            tmp_y = mat_y(:,1:10:2000);
        end
        num_f=1;
        if numel(size(result{1}))==4
            num_f = size(result{1},1);
        end
        if num_f==1
            yhat = conv(tmp_x,reshape(permute(result{1}(end:-1:1,:,:),[3 2 1]),1,[]),'valid')+result{1,2};
        else
            num_d = size(tmp_x,2);
            yhat = zeros([num_f,num_d]);
            for i=1:num_f
                yhat(i,:) = conv2(tmp_x,reshape(permute(squeeze(result{1,1}(i,end:-1:1,:,:)),[3 2 1]),[],1),'valid')+result{1,2}(i);
            end
        end
        yhat(yhat<0) = 0;
        yhat = (yhat'*result{2,1}+result{2,2})';
        mean((yhat-tmp_y).^2)
        plot(tmp_y),hold on,plot(yhat,'r-')
end







