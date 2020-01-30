% initialze weights given as inputs
function weights = init_weights(weights, prob, rng)

mask = rand(size(weights)) < prob;
weights = (2.0 * rand(size(weights)) - 1.0) .* mask;
if rng >= 0, 
    weights = weights .* rng;
else 
    weights(weights < 0) = rng; 
    weights(weights > 0) = -rng;
end; 