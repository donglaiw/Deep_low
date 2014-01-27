%{

addpath('../util');init
PP =pwd;
system([VLIB 'Para/p_run.sh 2 1 7 100 "' PP '" "B_dn(1,"'  ' ");" "' [PP R_DIR] 'bdn_1_"' ' ".mat"'])
system([VLIB 'Para/p_run.sh 2 1 1 7 "' PP '" "B_dn(2,"'  ' ");" "' [PP R_DIR] 'bdn_2_"' ' ".mat"'])



load data/test/berk_test_bk
for id =1:numel(Ins)
Ins{id} = single(U_thres(Ins{id},255,0));
end
save data/test/berk_test Is Ins

%}
function B_dn(did,id)
addpath('../util');init
switch did
case 1 
load([TT_DIR 'berk_test'])
nn=['1_' num2str(id-1)];
case 2 
load([TT_DIR 'pop_test'])
nn=['2_' num2str(id-1)];
end

err= zeros(6,2);
yhats= cell(1,6);
In = double(U_thres(Ins{id},255,0));
I = single(Is{id});
psz = 17;
sz = size(I);


  weightsSig = 2;
  mid = ceil(psz/2);
  sig = floor(psz/2)/weightsSig;
  for i=1:psz
    for j=1:psz
      d = sqrt((i-mid)^2 + (j-mid)^2);    
      pixel_weights(i,j) = exp((-d^2)/(2*(sig^2))) / (sig*sqrt(2*pi));
    end
  end
  pixel_weights = pixel_weights/max(pixel_weights(:));

pp = im2col(I,[psz psz])/255;
nps = (im2col(In,[psz psz])/255-0.5)/0.2;

nns={'0_1_1000_289_0_1000','0_3_1000_1000_289_0_1000','0_5_2000_2000_2000_2000_289_0_1000'};
sig = 25;
sz2 = sz-psz+1;



% patchify

pid=reshape(1:prod(sz2),sz2([2 1]))';
ppid = reshape(1:psz^2,[psz psz])';


for algo = 1:6
    disp(sprintf('algo: %d',algo));
    if algo<=3
        switch algo    
            case 1
                % optimal L2
                load init_p-1
                phat = ([single(nps);ones(1,size(nps,2))]'*[param{1};param{2}]*0.2+0.5)';
                yhat = 255*scol2im(phat,psz,sz(1),sz(2),'average');
                % initial test image error: 0.1715
                %{
                    ps = im2col(I,[psz psz]);
                    mean(sum(((single(nps) - single(ps))/255).^2,1))
                    mean(sum((single(phat)/255-single(ps)'/255).^2,2))
                %}
            case 2
                addpath([VLIB 'Low/Denoise/BM3D'])
                [~, yhat] = BM3D(1, double(In)/255, sig, 'np',0);
                phat = nps*0.2+0.5;
                yhat = yhat*255;
            case 3
                addpath([VLIB 'DeepL/dn_mlp'])
                model = {};
                model.weightsSig = 2;
                model.step = 1;
                fprintf('Starting to denoise...\n');
                [yhat,phat] = fdenoiseNeural(In, 25, model);        
                phat = phat(:,pid(:));
            end
    else
        fn = [R_DIR nns{algo-3} '/dl_r' nn '.mat'];
        if exist(fn,'file')
            load(fn) 
            phat = result{1}'*0.2+0.5;
            tmp_r = bsxfun(@times,phat,single(pixel_weights(:)));
            phat = phat(ppid(:),pid(:));
            yhat=scol2im(tmp_r,psz,sz(2),sz(1),'average')';
            what=scol2im(repmat(pixel_weights(:),1,size(phat,2)),psz,sz(2),sz(1),'average')';
            yhat=yhat./what;
            yhat = U_thres(yhat,1,0)*255;
            %yhat=scol2im(phat,psz,sz(2),sz(1),'average')';
        else
            error([fn ': no exist'])
        end        
    end
    err(algo,1) = U_psnr(I,yhat);
    err(algo,2) = mean(sum((pp-phat).^2));
    yhats{algo} = yhat;
end
err = err([1 4:6 3 2],:);
yhats = yhats([1 4:6 3 2]);
save([R_DIR 'bdn_' num2str(did) '_' num2str(id)],'yhats','err')
end
