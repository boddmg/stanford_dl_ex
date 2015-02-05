function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

pred_prob = [1:10];
%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
grad = zeros(size(theta));
gradStack = params2stack(grad, ei);
nl = numHidden+2;

m = size(data,2);
labels_number = size(stack{numHidden+1}.W,1);
%% forward prop
%%% YOUR CODE HERE %%%
last_output = cell(nl,1);
last_output{1} = data;
for i=1:(numHidden+1)
    temp_output = stack{i}.W * last_output{i} + repmat(stack{i}.b, 1, m);
    if i+1 < nl
        last_output{i+1} = f(temp_output,ei);
    else
        last_output{i+1} = temp_output; % It make the convergence become quick. 
    end
end
forward_prop_output = last_output;
h = exp(last_output{nl});
p = h ./ repmat(sum(h,1),size(h,1),1);
pred_prob = p;

%% return here if only predictions desired.
if po
  cost = -1; % ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
[k,p] = x_y_to_k_p(forward_prop_output{nl},labels');
cost = -sum(sum(k .* log(p)))/m;

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
gradStack = initialize_weights(ei);
delta = cell(numHidden+1,1);
calculated_output = forward_prop_output{numHidden+2};
true_output = labels';
[k,p] = x_y_to_k_p(calculated_output,true_output);
layer = numHidden+2;
delta{layer} = -(k-p);

for l=(nl-1):-1:2
    layer = layer - 1;
    delta{l} = (stack{l}.W' * delta{l+1}) .* forward_prop_output{l} .* (1-forward_prop_output{l});
end

for l=1:(numHidden+1)
    gradStack{l}.W = (delta{l+1} * forward_prop_output{l}')/m;
    gradStack{l}.b = mean(delta{l+1},2);
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
    for l=1:nl-1
        cost = cost + (ei.lambda/(2*m)) * sum(sum(stack{l}.W.^2));
        gradStack{l}.W = gradStack{l}.W + (ei.lambda/m)*stack{l}.W;
    end
    

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end 

function [k,p]=x_y_to_k_p(x,y)
num_samples = size(x,2);
num_theta = size(x,1);
p = exp(x);
p = p./repmat(sum(p,1),num_theta,1);
k = repmat( [1:num_theta]' , 1, num_samples);
k = bsxfun(@eq, k, repmat(y,num_theta,1));
end


function A = f(Z, ei)
    switch ei.activation_fun
        case 'logistic'
            A = 1 ./ (1+exp(-Z));
        case 'tanh'
            A = tanh(Z);
        otherwise
            A = Z;
    end
end

function A = fderivative(Z, ei)
    switch ei.activation_fun
        case 'logistic'
            A = 1 ./ (1+exp(-Z));
        case 'tanh'
            A = tanh(Z);
        otherwise
            A = Z;
    end
end

