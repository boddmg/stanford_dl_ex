%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);
norm_l2 = L1Norm(W*x);

%%% YOUR CODE HERE %%%
cost = params.lambda*sum(norm_l2) + sum(sum((W'*W*x-x).^2))*0.5;
Wgrad = params.lambda*W*x*bsxfun(@rdivide, x, norm_l2)' ...
    + W*(W'*W*x-x)*x' + (W*x)*(W'*W*x-x)';

% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);
end

function y = L1Norm(x)
    epsilon = 0.01;
    y = sqrt(sum(x.^2,1) + epsilon);
end