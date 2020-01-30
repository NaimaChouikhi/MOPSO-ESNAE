function precision=ClassAcc(rep,hidden)
rand('state', sum(100*clock));
don=xlsread('Breast.xlsx');
IPP=don(:,1:9)';
[dim1 dim2]=size(IPP);  
[dim1 dim2]=size(IPP);
% for i=1:dim1
%     for j=1:dim2
%         IPP(i,j)=IPP(i,j)/10;
%     end
% end
IUC = 9;
HUC = rep(1);
OUC = 9;
probInp  = rep(2);
rngInp   = [  1.00 ]; 
probRec  = rep(3);
rngRec   = [ -0.6 ];
probBack = rep(4);
rngBack  = [ -0.0];
w_rec = zeros(HUC, HUC, length(probRec));
w_in = zeros(HUC, IUC, length(probInp));
w_rec=reshape(hidden,HUC,HUC);
w_back = zeros(HUC, OUC, length(probBack));
for d=(1:length(probInp))
    w_in(:,:,d) = init_weights(w_in(:,:,d), probInp(d),rngInp(d));
end;

for d=(1:length(probBack))
    w_back(:,:,d) = init_weights(w_back(:,:,d), probBack(d),rngBack(d));
end;
IP=IPP;
TP=IPP;
x = zeros(HUC,size(TP,2));
x(:,1) = rand(1,HUC);
siz2=size(TP,2);
x2 = zeros(HUC,siz2);
x2(:,1) = rand(1,HUC);
w_out=rand(OUC,HUC);
iter = 100;
ojaC = 0.01;
lr = 0.001;
for t=2:size(TP,2), %run without any learning/training in reservoir and readout unit
    
    x(:,t) = tanh(w_in*IP(:,t) + w_rec*x(:,t-1));
    
end
%plot(x);
w_out = TP(:,20:end)*pinv(x(:,20:end));
%w_in=w_out';
for t=2:size(TP,2), %run without any learning/training in reservoir and readout unit
    
     x(:,t) = tanh(w_in*IP(:,t) + w_rec*x(:,t-1));
     y(:,t) = w_out*x(:,t);
end
rmsetrain = eval_mse(y(:,20:end),TP(:,20:end)) ;
IP=x(:,1:500)';
TP=don(1:500,10);
IPT=x(:,501:end)';
TPT=don(501:end,10);
svmStruct = svmtrain(IP,TP,'showplot',true);
Group = svmclassify(svmStruct,IPT);
SVMStruct1 = svmtrain(IP,TP);
Group1 = svmclassify(SVMStruct1,IPT);
RD=0; 
for(z=1:size(TPT,1));
if (Group1(z,:) == TPT(z,:))
    RD=RD+1;
end
end
  precision= RD/size(TPT,1);