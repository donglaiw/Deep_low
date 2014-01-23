% evaluate
addpath('../util');init
addpath([VLIB 'Mid/Boundary/Bd_eval'])
if ~exist('gts','var')
    load data/test/dn_ucb
end

if ~exist('tid','var')
tid=1;
end

num = numel(gts);
switch tid
    case 1
        % st: 0.7178,0.7419
        tmp=load('st_bd');
        pb = tmp.Es;
    case 2
        % python: 0.4725  0.3539
        tmp = load('500_300_151.mat');
        pb=tmp.pb;
end
num = 200;
roc = zeros(1,num);
thres= 0:0.05:1;
re = zeros(num,numel(thres),4);
%for i=1:num     
parfor i=1:num     
    [roc(i),re(i,:,:)] = U_occ(double(pb{i}),gts{i},thres);
end

re_m = squeeze(mean(re,1));
r = re_m(:,2)./(re_m(:,1)+(re_m(:,1)==0));
p = re_m(:,4)./(re_m(:,3)+(re_m(:,3)==0));
bid = r+p==0; r(bid)=[]; p(bid)=[];
f = 2.*r.*p./(r+p+((r+p)==0));
[max(f),U_roc(r,p)]


