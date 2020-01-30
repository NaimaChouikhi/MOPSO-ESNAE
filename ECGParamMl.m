function  z=ECGParamMl(HiddenWeight,params)
%rand('state', 1234);
rand('state', sum(100*clock))
donTr=load('../data/ECG200_TRAIN');
donTs=load('../data/ECG200_TEST');
don=[donTr(:,2:end);donTs(:,2:end)];
don=don';
donT=[donTr(:,1);donTs(:,1)];  
IPP=don;
% don=xlsread('Breast.xlsx');
% IPP=don(:,1:9)';
 %IPP = awgn(IPP,10,'measured');
TPP=IPP;
size(TPP);

IUC = 96;
%HUC = 20;
OUC = 96;
% unit counts (input, hidden, output)
probInp  = params(3);
rngInp   = [  1.00 ]; 
probRec  = params(4);
rngRec   = [ -0.6 ];
probBack = params(5);
rngBack  = [ 0.1];
probInter = params(6);
rngInter  = [ 0.1];
probRec2 = params(7);
rngRec2  = [ 0.1];
HUC=params(1);
HUC2=params(2);
A=HUC*HUC;
A2=HUC2*HUC2;
B=HUC*IUC;
B2=HUC2*HUC;
w_rec=reshape(HiddenWeight(1:A),HUC,HUC);
w_rec2 = reshape(HiddenWeight(A+1:A2+A),HUC2,HUC2);
w_in =reshape(HiddenWeight(A2+A+1:B+A2+A),HUC,IUC);
w_inter = reshape(HiddenWeight(B+A2+A+1:end),HUC2,HUC);
w_back = zeros(HUC, OUC, length(probBack));
for d=(1:length(probInp))
    w_in(:,:,d) = init_weights(w_in(:,:,d), probInp(d),rngInp(d));
end;
for d=(1:length(probBack))
    w_back(:,:,d) = init_weights(w_back(:,:,d), probBack(d),rngBack(d));
end;
for d=(1:length(probRec2))
    w_rec2(:,:,d) = init_weights(w_rec2(:,:,d), probRec2(d),rngRec2(d));
end;
for d=(1:length(probInter))
    w_inter(:,:,d) = init_weights(w_inter(:,:,d), probInter(d),rngInter(d));
end;
SpecRad = max(abs(eig(w_rec(:,:,1))));
if SpecRad>0,
    w_rec = w_rec ./ SpecRad;
end
SpecRad2 = max(abs(eig(w_rec2(:,:,1))));
if SpecRad2>0,
    w_rec2 = w_rec2 ./ SpecRad2;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IP=IPP;
TP=IPP;
% IPT=IPP(:,501:end);
% TPT=IPP(:,501:end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = zeros(HUC,size(TPP,2));
x(:,1) = rand(1,HUC);
w_out=rand(OUC,HUC);
siz2=size(TPP,2);
x2 = zeros(HUC2,siz2);
x2(:,1) = rand(1,HUC2);
iter = 5;
ojaC = 0.01;
lr = 0.00001;
for t=2:size(TPP,2), %run without any learning/training in reservoir and readout unit
    
      x(:,t) = tanh(w_in*IPP(:,t) + w_rec*x(:,t-1));
    x2(:,t) = tansig(w_rec2*x2(:,t-1) + w_inter*x(:,t));
    
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

%Calculate the error value for testing phase

f3=rmsetrain;
z=[f3];