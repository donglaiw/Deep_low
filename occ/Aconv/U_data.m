% 1D edge in 2D image
eid =0;

switch eid
case 0
    num_train =1e4;
    psz = 17;
    train_im = zeros([psz psz num_train],'single');
    train_bd = zeros([psz psz num_train],'single');
    did =0;
    switch did
    case 0
        pos = 1+ceil(rand(1,num_train)*15);
        cc = rand(2,num_train);
        for i=1:num_train
            train_im(:,1:pos(i),i) = cc(1,i);
            train_im(:,(1+pos(i)):end,i) = cc(2,i);
            train_bd(:,pos(i)+(0:1),i) = 1;
        end
        train_im = reshape(train_im,psz^2,num_train);
        train_bd = reshape(train_bd,psz^2,num_train);
    end
end
save(['train_conv' num2str(eid) '_' num2str(did)],'train_im','train_bd')
