function mat_xf = U_conv_feat(data,opt)
num = size(data.mat_x,2);
switch opt
case 1
    addpath('/data/vision/billf/stereo-vision/VisionLib/Donglai/DeepL/Deep_wrapper/decaf')
    mat_xf = ones([97 num],'single');
    mat_xf1 = zeros([3 num],'single');
    ff= U_f_LM(11);
    for i=1:48
        for j=1:3
            mat_xf1(j,:) = conv2(data.mat_x(121*(j-1)+(1:121),:),reshape(ff(:,:,i),[],1),'valid');
        end
        mat_xf(i*2-1,:) = max(mat_xf1,[],1); 
        mat_xf(i*2,:) = -min(mat_xf1,[],1); 
    end
    mat_xf(mat_xf<0) = 0;

end
