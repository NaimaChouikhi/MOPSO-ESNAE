%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPEA121
% Project Title: Multi-Objective Particle Swarm Optimization (MOPSO)
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

function PlotCosts(pop,rep)

    pop_costs=[pop.Cost];
    plot3(pop_costs(1,:),pop_costs(2,:),pop_costs(3,:),'b*');
      %plot(pop_costs(1,:),pop_costs(2,:),'bo');
    hold on;
    
    rep_costs=[F1.Cost];
    plot3(rep_costs(1,:),rep_costs(2,:),rep_costs(3,:),'r*');
    %plot(rep_costs(1,:),rep_costs(2,:),'r*');
    xlabel('1^{st} Objective: Reservoir size ');
    ylabel('2^{nd} Objective: Reservoir connectivity rate');
    zlabel('3^{rd} Objective: Accuracy(RMSE) ');
    grid on;hold off;
    
        pop_costs=[pop.Cost];
    plot3(pop_costs(1,:),pop_costs(2,:),pop_costs(3,:),'b*');
      %plot(pop_costs(1,:),pop_costs(2,:),'bo');
    hold on;
    
    rep_costs=[F1.Cost];
    plot3(rep_costs(1,:),rep_costs(2,:),rep_costs(3,:),'b*');
    F1.Cost=[BestSol.Cost];
    newrep_costs=[F1.Cost];
    plot3(newrep_costs(1,:),newrep_costs(2,:),newrep_costs(3,:),'r*');

    %plot(rep_costs(1,:),rep_costs(2,:),'r*');
    xlabel('1^{st} Objective: Reservoir size ');
    ylabel('2^{nd} Objective: Reservoir connectivity rate');
    zlabel('3^{rd} Objective: Accuracy(RMSE) ');
    grid on;
    
    
    
    hold off;

end
rep_costs=[rep.Cost];
    plot3(rep_costs(1,:),rep_costs(2,:),rep_costs(3,:),'b*');
    hold on
    rep_costs(3,:)=[BestSol.Cost];
    newrep_costs= rep_costs;
    plot3(newrep_costs(1,:),newrep_costs(2,:),newrep_costs(3,:),'r*');

    %plot(rep_costs(1,:),rep_costs(2,:),'r*');
    xlabel('1^{st} Objective: Reservoir size ');
    ylabel('2^{nd} Objective: Reservoir connectivity rate');
    zlabel('3^{rd} Objective: Accuracy(RMSE) ');
    grid on;
    hold off;
    hold on;
    pop_costs=[pop.Cost];
    plot(pop_costs(1,:),pop_costs(2,:),'b*');
    rep_costs=[rep.Cost];
    plot(rep_costs(1,:),rep_costs(2,:),'r*');
    %plot(rep_costs(1,:),rep_costs(2,:),'r*');
    xlabel('1^{st} Objective: Reservoir size ');
    ylabel('2^{nd} Objective: Reservoir connectivity rate');
    grid on;
    hold off;