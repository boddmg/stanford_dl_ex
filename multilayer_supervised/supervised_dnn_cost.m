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
gradStack = cell(numHidden+1, 1);
data_samples = size(data,2);
labels_number = size(stack{numHidden+1}.W,1);
%% forward prop
%%% YOUR CODE HERE %%%
fprintf('data size:%d,%d\r\n',size(data,1),size(data,2));
fprintf('labels size:%d,%d\r\n',size(labels,1),size(labels,2));
fprintf('stack{1}.W size:%d,%d\r\n',size(stack{1}.W,1),size(stack{1}.W,2));
fprintf('stack{1}.b size:%d,%d\r\n',size(stack{1}.b,1),size(stack{1}.b,2));
fprintf('stack{2}.W size:%d,%d\r\n',size(stack{2}.W,1),size(stack{2}.W,2));
fprintf('stack{2}.b size:%d,%d\r\n',size(stack{2}.b,1),size(stack{2}.b,2));
last_output = cell(numHidden+2,1);
last_output{1} = data;
for i=1:(numHidden+1)
    temp_output = stack{i}.W * last_output{i};
    temp_output = temp_output + repmat(stack{i}.b, 1, data_samples);
    if ei.activation_fun == 'logistic'
        last_output{i+1} = 1./(1+exp(temp_output));
    end
end
forward_prop_output = last_output;
fprintf('last_output size:%d,%d\r\n',size(last_output,1),size(last_output,2));

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
[k,p] = x_y_to_k_p(forward_prop_output{numHidden+2},labels');
cost = sum(sum(k .* log(p)));
%ceCost = forward_prop_output

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
gradStack = initialize_weights(ei);
last_x = forward_prop_output{numHidden+2};
last_y = labels';
for i=1:(numHidden+1)
    layer = numHidden+2-i;
    [k,p] = x_y_to_k_p(last_x,last_y);
    delta = -sum(k-p,2);
    gradStack{layer}.W = delta;
    gradStack{layer}.b = delta;
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%


%% reshape gradients into vector
[grad] = stack2params(gradStack);
cost = 1;
end

function [k,p]=x_y_to_k_p(x,y)
num_samples = size(x,2);
num_theta = size(x,1);
p = exp(x);
p = p./repmat(sum(p,1),num_theta,1);
k = repmat( [1:num_theta]' , 1, num_samples);
k = bsxfun(@eq, k, repmat(y,num_theta,1));
end



