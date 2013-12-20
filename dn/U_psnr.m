function y=U_psnr(im1,im2)
if max(im1(:))>1
    im1 = double(im1)/255;
end
if max(im2(:))>1
    im2 = double(im2)/255;
end
y = 10*log10(1/mean((double(im1(:))-double(im2(:))).^2));
