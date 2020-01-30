# Hierarchical multi-obejctive optimization of ESN-RAE
•	Echo State Network is a recurrent randomized architecture in which the hidden states are implemented by a non-trained recurrent hidden layer.

•	ESN is used as a recurrent Autoencoder (ESN-RAE) where the inputs are equal to the outputs.

•	Both of basic and Multi-layer ESNs are used as Autoencoders.

•	ESN (basic and ML) has some limits especially related to the setting of its architecture and some weights parameters.

•	Providing an ”optimal” reservoir(s) for a given problem is crucial.

•	Setting an optimal architecture while maintaining the complexity at a low level are the challenges to be led off.

•	A hierarchical bi-level evolutionary optimization based on Particle Swarm Optimization (PSO) is applied to  basic and ML ESN-RAEs.

•	The hidden activations of the best ESN-RAE are picked out to replace the original data.

•	The new data representation is squirted into an SVM classifier.

> ESN-RAE's architecture and weights optimization

A multi-objective evolutionary optimization (MOPSO) of basic and ML ESN-RAEs’ architectures.

The non-dominated solutions (basic and ML ESN-RAEs) obtained after MOPSO processing undergo a mono-objective evolutionary optimization (PSO) of their untrained weights. 


## Code, Data, Results
To run the code please execute main.m.

In this version, the ML-ESN-AE's initial parameters are altered to enhance the classification accuracy.