function [outdata, out_param] = classify_prob_test(varargin)
%CLASSIFYSVM Classify with libSVM an image
%
%		[outdata, out_param] = classify_svm(img, train, opt)
%
% INPUT
%   img    Multispectral image to be classified.
%   train  Training set image (zero is unclassified and will not be
%           considered).
%   opt    input parameters. Structure with each field correspondent to a
%           libsvm parameter
%           Below the availabel fields. The letters in the brackets corresponds to the flags used in libsvm:
%             "svm_type":	(-s) set type of SVM (default 0)
%                   0 -- C-SVC
%                   1 -- nu-SVC
%     	            2 -- one-class SVM
%     	            3 -- epsilon-SVR
%     	            4 -- nu-SVR
%             "kernel_type": (-t) set type of kernel function (default 2)
%                   0 -- linear: u'*v
%                   1 -- polynomial: (gamma*u'*v + coef0)^degree
%                   2 -- radial basis function: exp(-gamma*|u-v|^2)
%                   3 -- sigmoid: tanh(gamma*u'*v + coef0)
%                   4 -- precomputed kernel (kernel values in training_instance_matrix)
%          3   "kernel_degree": (-d) set degree in kernel function (default 3)
%             "gamma": set gamma in kernel function (default 1/k, k=number of features)
%           1  "coef0": (-r) set coef0 in kernel function (default 0)     1
%             "cost": (-c) set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
%             "nu": (-n) parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
%             "epsilon_regr": (-p) set the epsilon in loss function of epsilon-SVR (default 0.1)
%             "chache": (-m) set cache memory size in MB (default 100)
%             "epsilon": (-e) set tolerance of termination criterion (default 0.001)
%             "shrinking": (-h) whether to use the shrinking heuristics, 0 or 1 (default 1)
%        1     "probability_estimates": (-b) whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
%             "weight": (-wi) set the parameter C of class i to weight*C, for C-SVC (default 1)
%             "nfold": (-v) n-fold cross validation mode
%             "quite": (-q) quiet mode (no outputs)
%           For setting other default values, modify generateLibSVMcmd.
%
% OUTPUT
%   outdata    Classified image
%   out_param  structure reports the values of the parameters
%
% DESCRIPTION
% This routine classify an image according to the training set provided
% with libsvm. By default, the data are scaled and normalized to have unit
% variance and zero mean for each band of the image. If the parameters
% defining the model of the svm (e.g., the cost C and gamma) are not
% provided, the function call the routin MODSEL and which optimizes the
% parameters. Once the model is trained the image is classified and is
% returned as output.
%
% SEE ALSO
% EPSSVM, MODSEL, GETDEFAULTPARAM_LIBSVM, GENERATELIBSVMCMD, GETPATTERNS

% Mauro Dalla Mura
% Remote Sensing Laboratory
% Dept. of Information Engineering and Computer Science
% University of Trento
% E-mail: dallamura@disi.unitn.it
% Web page: http://www.disi.unitn.it/rslab

% Parse inputs
if nargin == 2
    data_set = varargin{1};
    train = varargin{2};
    in_param = struct;
%    in_param.kernel_type = 2;   % default RBF
elseif nargin == 3
    data_set = varargin{1};
    train = varargin{2};
    in_param = varargin{3};
end

% Default Parameters - Scaling the data
scaling_range = true;       % Scale each feature of the data in the range [-1,1]
scaling_std = true;         % Scale each feature of the data in order to have std=1

% Read in_param
if (isfield(in_param, 'scaling_range'))
    scaling_range = in_param.scaling_range;       % scaling_range
else
    in_param.scaling_range = scaling_range;
end
if (isfield(in_param, 'scaling_std'))
    scaling_std = in_param.scaling_std;           % scaling_range
else
    in_param.scaling_std = scaling_std;
end
% ------------------------

[nrows ncols nfeats] = size(data_set);
for i = 1: nfeats
    temp_y = reshape(data_set(:,:,i),1,[]);
    mm = mean(temp_y, 2);
    sdd = std(temp_y, 1, 2);
    data_set(:,:,i) = (data_set(:,:,i) - mm) / (2*sdd);
end
Ximg = double(reshape(data_set, nrows*ncols, nfeats));

% Transform training set in a format compliant to RF
[X, L] = getPatterns(data_set, train);
nclasses = length(unique(L));

% ========= Preprocessing =========

tic
% Train the model
for i = 1: nfeats
    temp_y = reshape(data_set(:,:,i),1,[]);
    mm = mean(temp_y, 2);
    sdd = std(temp_y, 1, 2);
    data_set(:,:,i) = (data_set(:,:,i) - mm) / (2*sdd);
end
Ximg = double(reshape(data_set, nrows*ncols, nfeats));
[X, L] = getPatterns(data_set, train);
% if ~(in_param.other.turning)
%     cmd = ['-q -s 1 -t 2 -nu' ' ' num2str(in_param.other.CCC) ' ' '-g' ' ' num2str(in_param.other.gamma) ' ' '-b 1']; % '-b 1'
%     model= svmtrain(double(L)',double(X)',cmd);
% else
%     accur_old = 0;
%     for ic = in_param.other.CCCiter
%         for ig = in_param.other.gammaiter
%             cmd = ['-q -s 1 -t 2 -nu' ' ' num2str(ic) ' ' '-g' ' ' num2str(ig) ' ' '-v 5'];
%             accur = svmtrain(double(L)',double(X)',cmd);
%             if accur > accur_old 
%                 opt_cmd = cmd(1:end-5);
%                 accur_old = accur;
%             end
%         end
%     end
%     out_param.other.opt_cmd = opt_cmd;
%     model = svmtrain(double(L)',double(X)',opt_cmd);
% end

model = in_param.model;

[predicted_labels, out_param.accuracy, out_param.prob_estimates] = svmpredict(ones(nrows*ncols, 1), Ximg, model, '-b 1');

outdata = predicted_labels;
out_param.time_tot = toc;