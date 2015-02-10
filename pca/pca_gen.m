%%================================================================
%% Step 0a: Load data
%  Here we provide the code to load natural image data into x.
%  x will be a 784 * 600000 matrix, where the kth column x(:, k) corresponds to
%  the raw image data from the kth 12x12 image patch sampled.
%  You do not need to change the code below.

addpath(genpath('../common'))
x = loadMNISTImages('../common/train-images-idx3-ubyte');
figure('name','Raw images');
randsel = randi(size(x,2),200,1); % A random selection of samples for visualization
display_network(x(:,randsel));

%%================================================================
%% Step 0b: Zero-mean the data (by row)
%  You can make use of the mean and repmat/bsxfun functions.

%%% YOUR CODE HERE %%%
IS_DEBUG = true;
if IS_DEBUG
    images = x(:,randsel);
else
    images = x;
end

images_dim_mean = mean(images,1);
images = bsxfun(@minus, images, images_dim_mean);


%%================================================================
%% Step 1a: Implement PCA to obtain xRot
%  Implement PCA to obtain xRot, the matrix in which the data is expressed
%  with respect to the eigenbasis of sigma, which is the matrix U.

%%% YOUR CODE HERE %%%
sigma = images * images' / size(images,2);
[U, S, V] = svd(sigma);
imagesRot = U' * images;


%%================================================================
%% Step 1b: Check your implementation of PCA
%  The covariance matrix for the data expressed with respect to the basis U
%  should be a diagonal matrix with non-zero entries only along the main
%  diagonal. We will verify this here.
%  Write code to compute the covariance matrix, covar. 
%  When visualised as an image, you should see a straight line across the
%  diagonal (non-zero entries) against a blue background (zero entries).

%%% YOUR CODE HERE %%%

% Visualise the covariance matrix. You should see a line across the
% diagonal against a blue background.
images_dim_mean = mean(imagesRot,1);
imagesRot = bsxfun(@minus, imagesRot, images_dim_mean);
covar = imagesRot * imagesRot' / size(imagesRot,2);
figure('name','Visualisation of covariance matrix');
imagesc(covar);

%%================================================================
%% Step 2: Find k, the number of components to retain
%  Write code to determine k, the number of components to retain in order
%  to retain at least 99% of the variance.

%%% YOUR CODE HERE %%%
singular_values = diag(S); 
retains = 0.90 * sum(singular_values);
temp_sum = 0;
for k = 1:size(singular_values)
    temp_sum = temp_sum + singular_values(k);
    if temp_sum >= retains
        break
    end
end

%%================================================================
%% Step 3: Implement PCA with dimension reduction
%  Now that you have found k, you can reduce the dimension of the data by
%  discarding the remaining dimensions. In this way, you can represent the
%  data in k dimensions instead of the original 144, which will save you
%  computational time when running learning algorithms on the reduced
%  representation.
% 
%  Following the dimension reduction, invert the PCA transformation to produce 
%  the matrix xHat, the dimension-reduced data with respect to the original basis.
%  Visualise the data and compare it to the raw data. You will observe that
%  there is little loss due to throwing away the principal components that
%  correspond to dimensions with low variation.

%%% YOUR CODE HERE %%%
imagesHat = U(:,1:k)' * images;
% U_invert = U;
% U_invert(:,k:end) = zeros(size(U,1),size(U,2)-k + 1);
% imagesHat = U_invert * images;
imagesHat = U(:,1:k) * imagesHat;

% Visualise the data, and compare it to the raw data
% You should observe that the raw and processed data are of comparable quality.
% For comparison, you may wish to generate a PCA reduced image which
% retains only 90% of the variance.

figure('name',['PCA processed images ',sprintf('(%d / %d dimensions)', k, size(x, 1)),'']);
display_network(imagesHat);
figure('name','Raw images');
display_network(images);

%%================================================================
%% Step 4a: Implement PCA with whitening and regularisation
%  Implement PCA with whitening and regularisation to produce the matrix
%  xPCAWhite. 

epsilon = 1e-1; 
%%% YOUR CODE HERE %%%
xPCAwhite = diag(1./sqrt(diag(S) + epsilon)) * U' * images;


%% Step 4b: Check your implementation of PCA whitening 
%  Check your implementation of PCA whitening with and without regularisation. 
%  PCA whitening without regularisation results a covariance matrix 
%  that is equal to the identity matrix. PCA whitening with regularisation
%  results in a covariance matrix with diagonal entries starting close to 
%  1 and gradually becoming smaller. We will verify these properties here.
%  Write code to compute the covariance matrix, covar. 
%
%  Without regularisation (set epsilon to 0 or close to 0), 
%  when visualised as an image, you should see a red line across the
%  diagonal (one entries) against a blue background (zero entries).
%  With regularisation, you should see a red line that slowly turns
%  blue across the diagonal, corresponding to the one entries slowly
%  becoming smaller.

%%% YOUR CODE HERE %%%
xPCAwhite_dim_mean = mean(xPCAwhite,1);
xPCAwhite = bsxfun(@minus, xPCAwhite, xPCAwhite_dim_mean);
covar = xPCAwhite * xPCAwhite' / size(xPCAwhite,2);

% Visualise the covariance matrix. You should see a red line across the
% diagonal against a blue background.
figure('name','Visualisation of covariance matrix');
imagesc(covar);

%%================================================================
%% Step 5: Implement ZCA whitening
%  Now implement ZCA whitening to produce the matrix xZCAWhite. 
%  Visualise the data and compare it to the raw data. You should observe
%  that whitening results in, among other things, enhanced edges.

%%% YOUR CODE HERE %%%
xZCAWhite = U * xPCAwhite;
% Visualise the data, and compare it to the raw data.
% You should observe that the whitened images have enhanced edges.
figure('name','ZCA whitened images');
display_network(xZCAWhite);
figure('name','Raw images');
display_network(x(:,randsel));
