function [model, out_param, kerneloption] = epsSVM_plain(instance_matrix,label_vector, in_param)
%EPSSVM Perform training of a svm
%
%		[model, out_param] = epsSVM(instance_matrix, label_vector, in_param)
%
% INPUT
%   instance_matrix:    matrix of the patterns used for training the model
%   label_vector:       vector with the labels of the patterns
%   in_param:           structure containing the optional parameters. For
%                       more information on the parameters please refer to generateLibSVMcmd.
%                       
% OUTPUT
%   model:              trained model
%   out_param:          same structure as in_param, with added some
%                       information
%                       
% DESCRIPTION
% This routine perform the training of a SVM on a training set. The values
% of the parameters not specified in in_param are the default ones defined
% in getDefaultParam_libSVM. If no value of the cost C (or gamma if the
% kernel type is non linear) then the value/s is optimized by modsel.
% The routine return the trained model and the structure with added
% information on the model selection.
%
% SEE ALSO
% GENERATELIBSVMCMD, MODSEL, GETDEFAULTPARAM_LIBSVM, CLASSIFY_SVM, GETPATTERNS

% $Id$

% Mauro Dalla Mura
% Remote Sensing Laboratory
% Dept. of Information Engineering and Computer Science
% University of Trento
% E-mail: dallamura@disi.unitn.it
% Web page: http://www.disi.unitn.it/rslab

% ------------------------
% % Default Parameters
gamma = [];         % default optimize it
cost = [];          % default optimize it

% check if c or g are defined, if so overwrite them
if (isfield(in_param, 'gamma'))
    gamma = in_param.gamma;       % epsilon SVM
end
if (isfield(in_param, 'cost'))
    cost = in_param.cost;       % epsilon SVM
end

% Pre-process data
instance_matrix2 = full(instance_matrix)';
[instance_matrix2,col_factor,temp] = unique(instance_matrix2','rows');  % Remove redundant patterns?
label_vector2 = label_vector(col_factor);

out_param = in_param;
out_param.isoptimized = false;

% switch out_param = in_param;

kernel_type = getDefaultParam_libSVM(in_param, 'kernel_type');


% if ((kernel_type ~= 0) && isempty(gamma)) || isempty(cost)  % OPTIMIZE C, Gamma
%     out_param.isoptimized = true;
% 
% %     [out_param] = modsel(label_vector2,instance_matrix2, out_param);
%     load out_param;
% end

% --- solve the problem ---
% fprintf('Starting LIBSVM\n');
% cmd = generateLibSVMcmd_mykernel(out_param, 'train');
% in_param.CCC
cmd = ['-t 4 -g 1 -c' ' ' num2str(in_param.other.CCC) '-b 1'];
% cmd
% cmd = '-t 4 -g 1 -c 15';

tic;
size(abs(max(instance_matrix(:))-min(instance_matrix(:))))
K1 = makekernel_two(instance_matrix,instance_matrix,abs(max(instance_matrix(:))-min(instance_matrix(:))),in_param.other.gamma);
K2 = makekernel_two(instance_matrixN,instance_matrixN,abs(max(instance_matrixN(:))-min(instance_matrixN(:))),in_param.other.gamma);
kerneloption = [abs(max(instance_matrix(:))-min(instance_matrix(:))) abs(max(instance_matrixN(:))-min(instance_matrixN(:))) ];

% summation
K = 0.4*K1+0.6*K2;%indianpines
K = [(1:size(K,1))' K];

model = svmtrain(label_vector,K,cmd);

% model = svmtrain(label_vector2, instance_matrix2, cmd);
% fprintf('Optimization finished in %3.2f sec\n',toc);


