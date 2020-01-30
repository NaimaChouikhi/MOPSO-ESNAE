function  z=ZDTECG(x,donTr,donTs)
rand('state', 1234);
% donTr=load('Earthquakes_TRAIN');
% donTs=load('Earthquakes_TEST');
don=[donTr(:,2:end);donTs(:,2:end)];
don=don';
% [dim1 dim2]=size(don);
% for i=1:dim1
%     for j=1:dim2
%         don(i,j)=don(i,j);
%     end
% end
donT=[donTr(:,1);donTs(:,1)];  
IPP=don;
%IPP = awgn(IPP,50,'measured');
TPP=IPP;
size(TPP);
[dim1 dim2]=size(IPP);

IUC = 96;
%HUC = 20;
OUC = 96;
% for i=1:dim1
%     for j=1:dim2
%         IPP(i,j)=IPP(i,j)/100;
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
rngBack  = [ -0.5];
HUC=x(1);
f2=x(1);
probRec=x(2);
probBack=x(3);
probInp=x(4);
f1=x(2);
w_in = zeros(HUC, IUC, length(probInp));
w_rec = zeros(HUC, HUC, length(probRec));
w_back = zeros(HUC, OUC, length(probBack));
for d=(1:length(probInp))
    w_in(:,:,d) = init_weights(w_in(:,:,d), probInp(d),rngInp(d));
end;

for d=(1:length(probRec))
    w_rec(:,:,d) = init_weights(w_rec(:,:,d), probRec(d),rngRec(d));
end;

for d=(1:length(probBack))
    w_back(:,:,d) = init_weights(w_back(:,:,d), probBack(d),rngBack(d));
end;
SpecRad = max(abs(eig(w_rec(:,:,1))));
if SpecRad>0,
    w_rec = w_rec ./ SpecRad;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IP=IPP;
TP=IPP;
% IPT=IPP(:,501:end);
% TPT=IPP(:,501:end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    
   % x(:,t) = tanh(w_in*IP(:,t) + w_rec*x(:,t-1));
     y(:,t) = w_out*x(:,t);
end
rmsetrain = eval_mse(y(:,10:end),TP(:,10:end)) ;

%Calculate the error value for testing phase

f3=rmsetrain;

z=[ f1
    f3];