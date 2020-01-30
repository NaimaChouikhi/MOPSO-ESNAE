%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPEA109
% Project Title: Implementation of Shuffled Frog Leaping Algorithm (SFLA)
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

function b = IsInRange(x, VarMin, VarMax)

    b = all(x>=VarMin) && all(x<=VarMax);

end