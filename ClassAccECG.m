function precision=ClassAccECG(rep,hidden,donTr,donTs)
rand('state', 1234);
% donTr=load('ECG200_TRAIN');
% donTs=load('ECG200_Test');
don=[donTr(:,2:end);donTs(:,2:end)];
don=don';
[dim1 dim2]=size(don);
% for i=1:dim1
%     for j=1:dim2
%         don(i,j)=don(i,j)/10;
%     end
% end
donT=[donTr(:,1);donTs(:,1)];  
IPP=don;
%IPP= awgn(IPP,50,'measured');
TPP=IPP;
size(TPP);
[dim1 dim2]=size(IPP);

% for i=1:dim1
%     for j=1:dim2
%         IPP(i,j)=IPP(i,j)/1;
%     end
% end
IUC = 96;
HUC = rep(1);
OUC = 96;
probInp  = rep(3);
rngInp   = [  1.00 ]; 
probRec  = rep(2);
rngRec   = [ -0.8 ];
probBack = rep(4);
rngBack  = [ -0.1];
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
for t=2:size(TP,2), %run without any learning/training in reservoir and readout unit
    
    x(:,t) = tanh(w_in*IP(:,t) + w_rec*x(:,t-1));
    
end
%plot(x);
w_out = TP(:,10:end)*pinv(x(:,10:end));
%w_in=w_out';
for t=2:size(TP,2), %run without any learning/training in reservoir and readout unit
    
     x(:,t) = tanh(w_in*IP(:,t) + w_rec*x(:,t-1));
     y(:,t) = w_out*x(:,t);
end
rmsetrain = eval_mse(y(:,10:end),TP(:,10:end)) ;
IP=x(:,1:100)';
TP=donT(1:100,:);
IPT=x(:,101:end)';
TPT=donT(101:end,:);
% GroupTrain=TP;  
% TrainingSet=IP;
% TestSet=IPT;
% u=unique(GroupTrain);
% numClasses=length(u);
% result = zeros(length(TestSet(:,1)),1);
% 
% %build models
% for k=1:numClasses
%     %Vectorized statement that binarizes Group
%     %where 1 is the current class and 0 is all other classes
%     G1vAll=(GroupTrain==u(k));
%     models(k) = svmtrain(TrainingSet,G1vAll);
% end
% %classify test cases
% for j=1:size(TestSet,1)
%     for k=1:numClasses
%         if(svmclassify(models(k),TestSet(j,:))) 
%             break;
%         end
%     end
%     result(j) = k;
% end
% RD=0; 
% for(z=1:size(TPT,1));
% if (result(z,:) == TPT(z,:))
%     RD=RD+1;
% end
% end
% precision= RD/size(TPT,1);
% % svmStruct = svmtrain(IP,TP,'showplot',true);
% % Group = svmclassify(svmStruct,IPT);
% % SVMStruct1 = svmtrain(IP,TP);
% % Group1 = svmclassify(SVMStruct1,IPT);
% % RD=0; 
% % for(z=1:size(TPT,1));
% % if (Group1(z,:) == TPT(z,:))
% %     RD=RD+1;
% % end
% % end
% %   precision= RD/size(TPT,1);
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