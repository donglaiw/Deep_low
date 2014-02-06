


if ~exist('did','var')
    did=1;
end
if ~exist('vv','var')
    vv=0;
end
switch did
case 1 
num=100;
nn='bdn_1_';
load([TT_DIR 'berk_test'])
case 2 
num=7;
nn='bdn_2_';
load([TT_DIR 'pop_test'])
end

if ~vv
    err1 = zeros(6,num);
    %err2 = zeros(6,num);
    for i=1:num
        try
            tmp = load(['result/' nn num2str(i)],'err');
            err1(:,i)= tmp.err(:,1);
            %err2(:,i)= tmp.err(:,2);
        catch
            i
        end
    end
   dlmwrite(['dn_err' num2str(did) '.txt'],err1','delimiter',',','precision','%.2f')
else
    for i=1:num
        tmp = load(['result/' nn num2str(i)],'yhats');
        %imwrite(uint8(Ins{i}),sprintf('result/dn/%d_%d_%d.png',did,i,0))
        for j=1:6
            imwrite(uint8(tmp.yhats{j}),sprintf('result/dn/%d_%d_%d.png',did,i,j))
        end
    end
end
