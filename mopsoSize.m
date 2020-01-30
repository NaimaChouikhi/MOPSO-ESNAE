clear all;

%% Problem Definition
donTr=load('../data/ECG200_TRAIN');
donTs=load('../data/ECG200_TEST');
% %donTr = awgn(donTr,10,'measured');
% %donTs = awgn(donTs,10,'measured');
% don=xlsread('Breast.xlsx');
% donnee=[ don(:,10) don(:,1:9)];
% donTr=donnee(1:100,1:10);
% donTs=donnee(101:end,1:10);
CostFunction=@(x) ZDTECG(x,donTr,donTs)     % Cost Function
nVar=4;             % Number of Decision Variables
ev=0;
nobj=2;
VarSize=[1 nVar];   % Size of Decision Variables Matrix

VarMinSize=10;          % Lower Bound of Variables
VarMaxSize=200;          % Upper Bound of Variables
VarMinProbRec=0.01;          % Lower Bound of Variables
VarMaxProbRec=1;
VarMinProbBack=0.01;          % Lower Bound of Variables
VarMaxProbBack=1; 
VarMinProbInp=0.01;          % Lower Bound of Variables
VarMaxProbInp=1; % Upper Bound of Variables
VarMin=[VarMinSize VarMinProbRec VarMinProbBack VarMinProbInp ];
VarMax=[VarMaxSize VarMaxProbRec VarMaxProbBack VarMaxProbInp];


%% MOPSO Parameters

MaxIt=50;           % Maximum Number of Iterations

nPop=20;            % Population Size

nRep=10;            % Repository Size

w=0.5;              % Inertia Weight
wdamp=0.99;         % Intertia Weight Damping Rate
c1=0.1;               % Personal Learning Coefficient
c2=0.2;               % Global Learning Coefficient

nGrid=7;            % Number of Grids per Dimension
alpha=0.2;          % Inflation Rate

beta=2;             % Leader Selection Pressure
gamma=2;            % Deletion Selection Pressure

mu=0.5;             % Mutation Rate

%% Initialization
%  phi1=2.05;
% % phi2=2.05;
% % phi=phi1+phi2;
% % chi=2/(phi-2+sqrt(phi^2-4*phi));
% % w=chi;          % Inertia Weight
% % %wdamp=1;        % Inertia Weight Damping Ratio
% % c1=chi*phi1;    % Personal Learning Coefficient
% % c2=chi*phi2;    % Global Learning Coefficient
% c2 =1.193;          % PSO parameter C1 
% c1 =1.193;        % PSO parameter C2 
% w =0.721; 
% % Velocity Limits
% VelMax=0.1*(VarMax-VarMin);
% VelMin=-VelMax;
empty_particle.Position=[];
empty_particle.Velocity=[];
empty_particle.Cost=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];
empty_particle.IsDominated=[];
empty_particle.GridIndex=[];
empty_particle.GridSubIndex=[];

pop=repmat(empty_particle,nPop,1);

for i=1:nPop
    
    pop(i).Position(1)=floor(unifrnd(VarMin(1),VarMax(1)));
    pop(i).Position(2)=unifrnd(VarMin(2),VarMax(2));
    pop(i).Position(3)=unifrnd(VarMin(3),VarMax(3));
    pop(i).Position(4)=unifrnd(VarMin(4),VarMax(4));
    pop(i).Velocity=zeros(VarSize);
    
    pop(i).Cost=CostFunction(pop(i).Position);
   
    ev=ev+1;
    % Update Personal Best
    pop(i).Best.Position=pop(i).Position;
    pop(i).Best.Cost=pop(i).Cost;
    
end
% Determine Domination
pop=DetermineDomination(pop);

rep=pop(~[pop.IsDominated]);

Grid=CreateGrid(rep,nGrid,alpha);

for i=1:numel(rep)
    rep(i)=FindGridIndex(rep(i),Grid);
end


%% MOPSO Main Loop

for it=1:MaxIt
    
    for i=1:nPop
        
        leader=SelectLeader(rep,beta);
        
        pop(i).Velocity = w*pop(i).Velocity ...
            +c1*rand(VarSize).*(pop(i).Best.Position-pop(i).Position) ...
            +c2*rand(VarSize).*(leader.Position-pop(i).Position);
        
        pop(i).Position = pop(i).Position + pop(i).Velocity;
        pop(i).Position(1)=floor(pop(i).Position(1));
        
        pop(i).Position(1) = max(pop(i).Position(1), VarMin(1));
        pop(i).Position(1) = min(pop(i).Position(1), VarMax(1));
        for(compt=2:nVar)
         pop(i).Position(compt) = max(pop(i).Position(compt), VarMin(compt));
        pop(i).Position(compt) = min(pop(i).Position(compt), VarMax(compt));
        end
        pop(i).Cost = CostFunction(pop(i).Position);
        ev=ev+1;
        % Apply Mutation
        pm=(1-(it-1)/(MaxIt-1))^(1/mu);
        if rand<pm
            NewSol.Position=Mutate(pop(i).Position,pm,VarMin,VarMax);
            NewSol.Position(1)=floor(pop(i).Position(1));
            NewSol.Cost=CostFunction(NewSol.Position);
            ev=ev+1;
            if Dominates(NewSol,pop(i))
                pop(i).Position=NewSol.Position;
                pop(i).Cost=NewSol.Cost;

            elseif Dominates(pop(i),NewSol)
                % Do Nothing

            else
                if rand<0.5
                    pop(i).Position=NewSol.Position;
                    pop(i).Cost=NewSol.Cost;
                end
            end
        end
        pop(i).Position(1)=floor(pop(i).Position(1));
        pop(i).Position(1) = max(pop(i).Position(1), VarMin(1));
        pop(i).Position(1) = min(pop(i).Position(1), VarMax(1));
        for(compt2=2:nVar)
         pop(i).Position(compt2) = max(pop(i).Position(compt2), VarMin(compt2));
        pop(i).Position(compt2) = min(pop(i).Position(compt2), VarMax(compt2));
        end
        if Dominates(pop(i),pop(i).Best)
            pop(i).Best.Position=pop(i).Position;
            pop(i).Best.Cost=pop(i).Cost;
            
        elseif Dominates(pop(i).Best,pop(i))
            % Do Nothing
            
        else
            if rand<0.5
                pop(i).Best.Position=pop(i).Position;
                pop(i).Best.Cost=pop(i).Cost;
            end
        end
        
    end
    
    % Add Non-Dominated Particles to REPOSITORY
    rep=[rep
         pop(~[pop.IsDominated])]; %#ok
    
    % Determine Domination of New Resository Members
    rep=DetermineDomination(rep);
    
    % Keep only Non-Dminated Memebrs in the Repository
    rep=rep(~[rep.IsDominated]);
    
    % Update Grid
    Grid=CreateGrid(rep,nGrid,alpha);

    % Update Grid Indices
    for i=1:numel(rep)
        rep(i)=FindGridIndex(rep(i),Grid);
    end
    
    % Check if Repository is Full
    if numel(rep)>nRep
        
        Extra=numel(rep)-nRep;
        for e=1:Extra
            rep=DeleteOneRepMemebr(rep,gamma);
        end
        
    end
    % Plot Costs
%     figure(1);
%     PlotCosts(pop,rep);
    
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Number of Repository Members = ' num2str(numel(rep))]);
    %pause;
    % Damping Inertia Weight
    w=w*wdamp;
%  pause  
end

%% Resluts
disp(' ');

EPC=[rep.Cost];
for j=1:(numel(rep))
   % if (j==4)
    if (j==3)
        break;
    end
    disp(['Objective #' num2str(j) ':']);
    disp(['      Min = ' num2str(min(EPC(j,:)))]);
    disp(['      Max = ' num2str(max(EPC(j,:)))]);
    disp(['    Range = ' num2str(max(EPC(j,:))-min(EPC(j,:)))]);
    disp(['    St.D. = ' num2str(std(EPC(j,:)))]);
    disp(['     Mean = ' num2str(mean(EPC(j,:)))]);nVar=4;            % Number of Decision Variables
end

BestSol1=[];
BestSol2=[];
for compt=1:numel(rep) 
    i=0;
compt
IUC = 96;
HUC = rep(compt).Position(1);
OUC = 96;

MaxIt=50;      % Maximum Number of Iterations

nPop=8;        % Population Size (Swarm Size)

probInp  = rep(compt).Position(3);
rngInp   = [  1.00 ]; 
probRec  = rep(compt).Position(2);
rngRec   = [ -0.8 ];
probBack = rep(compt).Position(4);
rngBack  = [ -0.5];
w_rec = zeros(HUC, HUC, length(probRec));
for d=(1:length(probRec))
    w_rec(:,:,d) = init_weights(w_rec(:,:,d), probRec(d),rngRec(d));
end;
SpecRad = max(abs(eig(w_rec(:,:,1))));
if SpecRad>0,
    w_rec = w_rec ./ SpecRad;
end
SpecRad;
%n = 6;      % Size of the swarm " no of birds "
func=6;
iter=0;
A=HUC*HUC;
nVar=A;
VarSize=[1 nVar];   % Size of Decision Variables Matrix
current_position=zeros(nVar,nPop);
partWeights=reshape(w_rec,A,1);
[n1,m1]=size(partWeights);
R1 = rand(n1, nPop);   % PSO randomness parameter R1
R2 = rand(n1, nPop);   % PSO randomness parameter R2
for (i=1:nPop)
    for d=(1:length(probRec))
    w_rec(:,:,d) = init_weights(w_rec(:,:,d), probRec(d),rngRec(d));
end;
     partWeights=reshape(w_rec,A,1);
     current_position(:,i)=partWeights;   % particles positions initialization from reservoir weights (equal in the beginning) 
     %current_position(:,i)=0.4-0.8*rand(n1, 1);
end    
 %[siz1,siz2]=size( current_position);  
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% PSO Optimization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          
VarMinRec=-1;          % Lower Bound of Variables
VarMaxRec=1;
VarMin=[ VarMinRec  ];
VarMax=[ VarMaxRec];

%% PSO Parameters

% PSO Parameters
w=0.9;            % Inertia Weight
%wdamp=0.99;     % Inertia Weight Damping Ratio
c1=0.1;         % Personal Learning Coefficient
c2=0.2;         % Global Learning Coefficient

% If you would like to use Constriction Coefficients for PSO,
% uncomment the following block and comment the above set of parameters.

% % Constriction Coefficients
%  phi1=2.05;
% phi2=2.05;
% phi=phi1+phi2;
% chi=2/(phi-2+sqrt(phi^2-4*phi));
% w=chi;          % Inertia Weight
% wdamp=1;        % Inertia Weight Damping Ratio
% c1=chi*phi1;    % Personal Learning Coefficient
% c2=chi*phi2;    % Global Learning Coefficient

% Velocity Limits
VelMax=0.1*(VarMax-VarMin);
VelMin=-VelMax;

%% Initialization

empty_particle.Position=[];
empty_particle.Cost=[];
empty_particle.Velocity=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];

particle=repmat(empty_particle,nPop,1);

GlobalBest.Cost=inf;

for i=1:nPop
 for ii=1:nVar 
    % Initialize Position
    %particle(i).Position=unifrnd(VarMin,VarMax,VarSize);
    particle(i).Position(ii)=current_position(ii,i);
 end
    % Initialize Velocity
    particle(i).Velocity=zeros(VarSize);
    
    % Evaluation
    particle(i).Cost=ECGParam(particle(i).Position,rep(compt).Position, donTr,donTs);
    
    % Update Personal Best
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
    
    % Update Global Best
    if particle(i).Best.Cost<GlobalBest.Cost
        
        GlobalBest=particle(i).Best;
        
    end
    
end

BestCost=zeros(MaxIt,1);

%% PSO Main Loop

for it=1:MaxIt
    
    for i=1:nPop
        
        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity ...
            +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            +c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
        
        % Apply Velocity Limits
        
        %particle(i).Velocity = max(particle(i).Velocity,VelMin);
        %particle(i).Velocity = min(particle(i).Velocity,VelMax);
%         for(compteur=1:nVar)
%          particle(i).Velocity(compteur) = max(particle(i).Velocity(compteur), VelMin(compteur));
%         particle(i).Velocity(compteur) = min(particle(i).Velocity(compteur), VelMax(compteur));
%         end
        
        % Update Position
        for ii=1:nVar 
        if (particle(i).Position(ii)~=0)
        particle(i).Position(ii) = particle(i).Position(ii) + particle(i).Velocity(ii);
        end
        end
        % Velocity Mirror Effect
       % IsOutside=(particle(i).Position<VarMin | particle(i).Position>VarMax);
       % particle(i).Velocity(IsOutside)=-particle(i).Velocity(IsOutside);
        
%         % Apply Position Limits
%         particle(i).Position = max(particle(i).Position,VarMin);
%         particle(i).Position = min(particle(i).Position,VarMax);
%   
%         for(compteur=1:nVar)
%          particle(i).Position(compteur) = max(particle(i).Position(compteur), VarMin(compteur));
%         particle(i).Position(compteur) = min(particle(i).Position(compteur), VarMax(compteur));
%         end
        % Evaluation
        particle(i).Cost = ECGParam(particle(i).Position,rep(compt).Position,donTr,donTs);
        
        % Update Personal Best
        if particle(i).Cost<particle(i).Best.Cost
            
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            
            % Update Global Best
            if particle(i).Best.Cost<GlobalBest.Cost
                
                GlobalBest=particle(i).Best;
                
            end
            
        end
        
    end
    
    BestCost(it)=GlobalBest.Cost;
    
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
   % w=w*wdamp;
    
end

BestSol1 = [BestSol1 GlobalBest];
preci1(compt)=ClassAccECG(rep(compt).Position,BestSol1(compt).Position, donTr,donTs);

   

end
fprintf('The classification accuracies of the non-dominated solutions after weights optimization are\n');
ClassAcc= preci1
[bestPar,g] = max(ClassAcc); 
fprintf('The best solution is the particle number %d with a classification accuracy equal to %f\n', g,bestPar);
figure(2)
pop_costs=[pop.Cost];
   plot(pop_costs(1,:),pop_costs(2,:),'b*');
    hold on
rep_costs=[rep.Cost];
   plot(rep_costs(1,:),rep_costs(2,:),'r*');
%rep_costs(2,:)=[BestSol1.Cost];
   % newrep_costs= rep_costs;
    %plot(newrep_costs(1,:),newrep_costs(2,:),'g*');
     title('Population (blue stars) convergence towards Pareto Front obtained after MOPSO (red stars)')
 xlabel('1^{st} Objective: Reservoir connectivity rate');
 ylabel('2^{nd} Objective: Accuracy(RMSE) ');
saveas(gcf, '../results/fig2.png')
print('../results/plot', '-dpdf')
    grid on;
    hold off;