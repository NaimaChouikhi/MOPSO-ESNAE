function weight = init_weight(prob, rng)

weight = 0;
if rand > prob, return; end;

weight = 2.0 * rand - 1.0;
if rng >= 0, 
    weight = weight .* rng;
else 
    if weight  < 0; weight =  rng;
    else weight = -rng; 
    end;
end; 