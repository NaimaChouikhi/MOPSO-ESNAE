function precision=ClassAccECGMl(rep,hidden)
%rand('state', 1234);
rand('state', sum(100*clock));
donTr=load('../data/ECG200_TRAIN');
donTs=load('../data/ECG200_TEST');
don=[donTr(:,2:end);donTs(:,2:end)];
don=don';
donT=[donTr(:,1);donTs(:,1)];  
IPP=don;
TPP=IPP;
% don=xlsread('Breast.xlsx');
% IPP=don(:,1:9)';
%IPP = awgn(IPP,10,'measured');
TPP=IPP;

size(TPP);
% for i=1:dim1
%     for j=1:dim2
%         IPP(i,j)=IPP(i,j)/10;
%     end
% end
IUC = 96;
HUC = rep(1);
HUC2 = rep(2);
OUC = 96;
probInp  = rep(5);
rngInp   = [  1.00 ]; 
probRec  = rep(3);
rngRec   = [ -0.6 ];
probBack = rep(4);
rngBack  = [ -0.0];
probInter = rep(6);
rngInter  = [ 0.1];
probRec2 = rep(7);
rngRec2  = [ 0.1];
A=HUC*HUC;
A2=HUC2*HUC2;
B=HUC*IUC;
B2=HUC2*HUC;
w_rec=reshape(hidden(1:A),HUC,HUC);
w_rec2 = reshape(hidden(A+1:A2+A),HUC2,HUC2);
w_in =reshape(hidden(A2+A+1:B+A2+A),HUC,IUC);
w_inter = reshape(hidden(B+A2+A+1:end),HUC2,HUC);
w_back = zeros(HUC, OUC, length(probBack));
for d=(1:length(probRec2))
    w_rec2(:,:,d) = init_weights(w_rec2(:,:,d), probRec2(d),rngRec2(d));
end;
for d=(1:length(probInter))
    w_inter(:,:,d) = init_weights(w_inter(:,:,d), probInter(d),rngInter(d));
end;
w_back = zeros(HUC, OUC, length(probBack));
for d=(1:length(probInp))
    w_in(:,:,d) = init_weights(w_in(:,:,d), probInp(d),rngInp(d));
end;

for d=(1:length(probBack))
    w_back(:,:,d) = init_weights(w_back(:,:,d), probBack(d),rngBack(d));
end;
IP=IPP;
TP=IPP;
x = zeros(HUC,size(TPP,2));
x(:,1) = rand(1,HUC);
w_out=rand(OUC,HUC);
siz2=size(TPP,2);
x2 = zeros(HUC2,siz2);
x2(:,1) = rand(1,HUC2);
for t=2:size(TPP,2), %run without any learning/training in reservoir and readout unit
    
      x(:,t) = tanh(w_in*IPP(:,t) + w_rec*x(:,t-1));
    x2(:,t) = tanh(w_rec2*x2(:,t-1) + w_inter*x(:,t));
    
end
%plot(x);
w_out = TPP(:,10:end)*pinv(x2(:,10:end));
%w_in=w_out';

for t=2:size(TPP,2), %run without any learning/training in reservoir and readout unit
    
     x(:,t) = tanh(w_in*IPP(:,t) + w_rec*x(:,t-1));
    x2(:,t) = tanh(w_rec2*x2(:,t-1) + w_inter*x(:,t));
    y(:,t)=w_out* x2(:,t);
    
end
rmsetrain = eval_mse(y(:,10:end),TP(:,10:end)) ;
% IP=x2(:,1:100)';
% TP=donT(1:100,:);
% IPT=x2(:,101:end)';
% TPT=donT(101:end,:);
IP=x2(:,1:100)';
TP=donT(1:100,:);
IPT=x2(:,101:end)';
TPT=donT(101:end,:);
svmStruct = svmtrain(IP,TP);
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