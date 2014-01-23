eid=0;
did=0;
mid = 1;
load(['train_conv' num2str(eid) '_' num2str(did)],'train_im','train_bd')

fsz = 5;
psz = 17;
num_train = size(train_im,2);
mat = reshape(train_im,[psz psz num_train]);
mat2 = zeros([psz-fsz+1 psz-fsz+1 num_train],'single');

switch mid
case 0
    W0 = zeros(fsz,'single');
    W0(:,2) = 1;
    W0(:,4) = -1;
    W0 = W0*100;

    for i=1:num_train
        mat2(:,:,i) = conv2(mat(:,:,i),W0,'valid');
    end
    sz0 = (psz-fsz+1); 
    sz0_2 = sz0^2; 
    b0 = zeros(sz0,sz0,'single');
    mat2_in = reshape(mat2,sz0_2,num_train)';
    mat3_in = tanh(abs(bsxfun(@plus,mat2_in,b0(:)')));

    U_cen = reshape(1:psz^2,psz,psz);
    U_cen = U_cen(3:end-2,3:end-2);
    %{
    %debug
    find(a>10,1,'first')
    %}
    err=sum((mat3_in - train_bd(U_cen(:),:)').^2,2);
    mean(err(1:10:end))
    param = cell(1,2);
    param{1,1} = reshape(W0,[1 1 fsz fsz]);
    param{1,2} = reshape(b0,[1 sz0 sz0]);
case 1
    W0 = zeros(fsz,'single');
    W0(:,2) = 1;
    W0(:,4) = -1;
    W0 = W0*100;
    for i=1:num_train
        mat2(:,:,i) = conv2(mat(:,:,i),W0,'valid');
    end
    sz0 = (psz-fsz+1); 
    sz0_2 = sz0^2; 
    b0 = zeros(sz0,sz0,'single');
    mat2_in = reshape(mat2,sz0_2,num_train)';
    mat3_in = tanh(abs(bsxfun(@plus,mat2_in,b0(:)')));

    U_cen = reshape(1:psz^2,psz,psz);
    U_cen = U_cen(3:end-2,3:end-2);
    %{
    %debug
    find(a>10,1,'first')
    %}
    err=sum((mat3_in - train_bd(U_cen(:),:)').^2,2);
    mean(err(1:10:end))
    param = cell(1,2);
    param{1,1} = reshape(W0',[1 1 fsz fsz]);
    param{1,2} = reshape(b0,[1 sz0 sz0]);
end
save(['init_p' num2str(mid)],'param')
