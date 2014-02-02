load ha

load ../data/train/conv_11_0

%{
% display
for i=1:36
    subplot(6,6,i),imagesc(uint8(255*10*squeeze(result{1}(i,1,:,:))))
    %subplot(6,6,i),imagesc(uint8(255*10*permute(squeeze(result{1}(i,:,:,:)),[2 3 1])))
end
   %} 
num=2000;
t_x = mat_x(:,1:num);
t_y = mat_y(:,1:num);
sz = size(result{1,1});
sz2 = size(t_x); 
filt = reshape(permute(result{1,1},[3 4 2 1]),sz2(1),[]);
num_filt = sz(1);

out_l1 = zeros(num_filt,num);
for i=1:num_filt
    out_l1(i,:) = conv2(t_x,flipud(filt(:,i)),'valid') + result{1,2}(i);
    %out_l1(i,:) = conv2(t_x,filt(:,i),'valid') + result{1,2}(i);
end

out_l2 = out_l1'*result{2,1}+result{2,2};



