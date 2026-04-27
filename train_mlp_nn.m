function layers = train_mlp_nn(inputDim, numHiddenUnits, numHiddenLayers, actFcnHidden)
%TRAIN_MLP_NN Define the MLP-NN architecture used in the paper.
%   layers = TRAIN_MLP_NN(inputDim, numHiddenUnits, numHiddenLayers, actFcnHidden)
%   returns a layer array with:
%       - Input layer of size inputDim
%       - numHiddenLayers fully-connected hidden layers with
%         numHiddenUnits neurons and tanh activation
%       - Output layer with 2 neurons (for [u, v]) with NO activation
%         (linear output for regression)

layers = [
    featureInputLayer(inputDim, 'Name','input')
];

for i = 1:numHiddenLayers
    layers = [
        layers
        fullyConnectedLayer(numHiddenUnits, 'Name',sprintf('fc_%d',i))
        tanhLayer('Name',sprintf('tanh_%d',i))
    ];
end

% CRITICAL: Output layer should have NO activation for regression
% The regressionLayer will compute MSE loss
layers = [
    layers
    fullyConnectedLayer(2, 'Name','fc_out')
    regressionLayer('Name','regressionoutput')
];

end
