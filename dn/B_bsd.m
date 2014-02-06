%{
PP =pwd;
system([VLIB 'Donglai/Para/p_run.sh 2 1 1 100 "' PP '" "B_bsd("'  ');" "' [PP R_DIR] 'bsd"' '".mat"'])
%}
function B_bsd(id)
init
load([T_DIR 'berk_test'])
err= zeros(1,5);
yhats= cell(1,5);
In = double(Ins{id});
I = Is{id};
psz = 17;
sz = size(I);
sig = 25;
for algo =1:6
    yhat = In;
    switch algo    
    case 1
        yhat = In;
    case 2
        % optimal L2
        load init_p-1
        nps = im2col(In,[psz psz]);        
        phat = [single(nps);ones(1,size(nps,2))]'*[param{1};param{2}];
        yhat = uint8(scol2im(phat',psz,sz(1),sz(2),'average'));
        % initial test image error: 0.1715
        %{
            ps = im2col(I,[psz psz]);
            mean(sum(((single(nps) - single(ps))/255).^2,1))
            mean(sum((single(phat)/255-single(ps)'/255).^2,2))
        %}
    case 3
         addpath([VLIB 'Low/Denoise/BM3D'])
         [~, yhat] = BM3D(1, double(In)/255, sig, 'np',0);
         yhat = yhat*255;
    case 4
        exp_id = 0; 
        model_id = 1;
        num_epoch =10000;
        nn = sprintf('%d_%d_%d',exp_id,model_id,num_epoch);
        fn = [R_DIR nn '/dl_r' nn '.mat'];
        if exist(fn,'file')
            load(fn) 
            yhat=scol2im(result{1}',psz,sz(1),sz(2),'average');
        end        
        yhat = yhat*255;
    case 5
        exp_id = 0; 
        model_id = 3;
        num_epoch = 1000;
        nn = sprintf('%d_%d_%d',exp_id,model_id,num_epoch);
        fn = [R_DIR nn '/' num2str(id-1) 'dl_r1_' nn '.mat'];
        if exist(fn,'file')
            load(fn) 
            yhat=scol2im(result{1}',psz,sz(1),sz(2),'average');
        end        
        yhat = yhat*255;
    case 6
        addpath([VLIB 'DeepL/dn_mlp'])
        % define some parameters for denoising
        model = {};
        % width of the Gaussian window for weighting output pixels
        model.weightsSig = 2;
        % the denoising stride. Smaller is better, but is computationally 
        % more expensive.
        model.step = 1;
        % denoise
        fprintf('Starting to denoise...\n');
        yhat = fdenoiseNeural(In, 25, model);        
    end
    err(algo) = U_psnr(I,yhat);
    yhats{algo} = algo;
end
        
save([R_DIR 'bsd_' num2str(id)],'yhats','err')
end
