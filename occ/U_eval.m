% evaluate
addpath('../util');init
addpath([VLIB 'Mid/Boundary/Bd_eval'])
addpath('~/Desktop/Edge/Hack')
if ~exist('gts','var')
    load data/test/dn_ucb
end

if ~exist('tid','var')
tid=1;
end

num = numel(gts);
switch tid
    case 1
        % st
        tmp=load('st_bd');
        pb = tmp.Es;
    case 2
        % python
        tmp = load('300_200_151.mat');
        if ~exist('tmp.pb','var')
            pb = tmp.pb;
            rr = tmp.rr;
            parfor i=1:num
                pb{i} = stToEdges( rr{i}, 1 );
            end
            save('300_200_151.mat','pb','rr')
        end
        pb=tmp.pb;
end
roc = zeros(1,num);
re = zeros(num,51,4);
parfor i=1:num     
    [roc(i),re(i,:,:)] = U_occ(pb{i},gts{i});
end

re_m = squeeze(mean(re,1));
r = re_m(:,2)./(re_m(:,1)+(re_m(:,1)==0));
p = re_m(:,4)./(re_m(:,3)+(re_m(:,3)==0));
bid = r+p==0; r(bid)=[]; p(bid)=[];
f = 2.*r.*p./(r+p+((r+p)==0));
[max(f),U_roc(r,p)]


