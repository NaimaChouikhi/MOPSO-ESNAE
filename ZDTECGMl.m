function  z=ZDTECGMl(x)
%rand('state', 1234)
rand('state', sum(100*clock));
donTr=load('../data/ECG200_TRAIN');
donTs=load('../data/ECG200_TEST');
don=[donTr(:,2:end);donTs(:,2:end)];
don=don';
donT=[donTr(:,1);donTs(:,1)];  
IPP=don;
% don=xlsread('Breast.xlsx');
IUC = 96;
%HUC = 20;
OUC = 96;
% IPP=don(:,1:9)';
% IPP = awgn(IPP,10,'measured');
TPP=IPP;
size(TPP);

% IUC = 96;
% %HUC = 20;
% OUC = 96;
% for i=1:dim1
%     for j=1:dim2
%         IPP(i,j)=IPP(i,j)/10;
%     end
% end
% don(:,1:9)=don(:,1:9)/100;
% don(:,10)=don(:,10)/10;
% unit counts (input, hidden, output)
%probInp  = [  1.00 ];
rngInp   = [  1.00 ]; 
%probRec  = [  0.0 ];
rngRec   = [ -0.6 ];
%probBack = [  0.0 ];
rngBack  = [ 0.1];
%probRec2  = [  0.005 ];
rngRec2   = [  -0.6 ]; 
%probInter = [  1 ];
rngInter  = [ 0.2];
HUC=x(1);
HUC2=x(2);
f2=(x(1)+x(2))/2;
probRec=x(3);
probBack=x(4);
probInp=x(5);
probInter=x(6);
probRec2=x(7);
f1=(x(3)+x(7))/2;
w_in = zeros(HUC, IUC, length(probInp));
w_rec = zeros(HUC, HUC, length(probRec));
w_rec2 = zeros(HUC2, HUC2, length(probRec2));
w_inter = zeros(HUC2, HUC, length(probInter));
w_back = zeros(HUC, OUC, length(probBack));
for d=(1:length(probInp))
    w_in(:,:,d) = init_weights(w_in(:,:,d), probInp(d),rngInp(d));
end;

for d=(1:length(probRec))
    w_rec(:,:,d) = init_weights(w_rec(:,:,d), probRec(d),rngRec(d));
end;
for d=(1:length(probRec2))
    w_rec2(:,:,d) = init_weights(w_rec2(:,:,d), probRec2(d),rngRec2(d));
end;
for d=(1:length(probInter))
    w_inter(:,:,d) = init_weights(w_inter(:,:,d), probInter(d),rngInter(d));
end;

for d=(1:length(probBack))
    w_back(:,:,d) = init_weights(w_back(:,:,d), probBack(d),rngBack(d));
end;
SpecRad = max(abs(eig(w_rec(:,:,1))));
if SpecRad>0,
    w_rec = w_rec ./ SpecRad;
end
SpecRad2= max(abs(eig(w_rec2(:,:,1))));
if SpecRad2>0,
    w_rec2 = w_rec2 ./ SpecRad;
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

%Calculate the error value for testing phase

f3=rmsetrain;

z=[ f1
    f3];