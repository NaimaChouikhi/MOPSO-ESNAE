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

function PlotCostsBiObj(pop,rep)

    pop_costs=[pop.Cost];
    %plot3(pop_costs(1,:),pop_costs(2,:),pop_costs(3,:),'bo');
      plot(pop_costs(1,:),pop_costs(2,:),'b*');
    hold on;
    
    rep_costs=[rep.Cost];
    %plot3(rep_costs(1,:),rep_costs(2,:),rep_costs(3,:),'r*');
    plot(rep_costs(1,:),rep_costs(2,:),'r*');
    xlabel('1^{st} Objective: Reservoir size ');
    ylabel('2^{nd} Objective: Accuracy');
    %zlabel('2^{nd} Objective: Accuracy(RMSE) ');
    grid on;
    
    hold off;

end