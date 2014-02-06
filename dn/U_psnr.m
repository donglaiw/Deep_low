function y=U_psnr(y,yhat)
if max(y(:))>1
    y = double(y)/255;
end
if max(yhat(:))>1
    yhat = double(yhat)/255;
end
yhat(yhat>1)=1;
yhat(yhat<0)=0;
y = 10*log10(1/mean((double(y(:))-double(yhat(:))).^2));
