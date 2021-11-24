t1=clock;
rng('default')
rng(1) %random seed
trial_num = 10; %number of trials, in the paper we used 10
overall_OA = []; overall_AA = []; overall_kappa = []; overall_CA = [];
overall_cmd = []; 
best_param = [];

[no_rows,no_lines, no_bands] = size(HSI);
img1=reshape(HSI,[no_rows*no_lines,no_bands]);
prediction_map = zeros(no_rows,no_lines,trial_num);
%% Select the number of training samples for each class
RandSampled_Num = repmat(pts_per_class,1,K_Known);

Nonzero_map = zeros(no_rows,no_lines);
Nonzero_index =  find(Y2d ~= 0);      %% find labeled pixels
Nonzero_map(Nonzero_index)=1;                     %% labeled pixels = 1

%% Create the experimental set based on groundtruth of HSI
Train_Label = [];
Train_index = [];
for ii = 1: K_Known
   index_ii =  find(Y2d == ii);
   class_ii = ones(length(index_ii),1)* ii;
   Train_Label = [Train_Label class_ii'];
   Train_index = [Train_index index_ii'];   
end
%%% Train_Label and Train_index are row vectors
%%% Train_Label, the indices are the vectorized location
trainall = zeros(2,length(Train_index));
trainall(1,:) = Train_index;
trainall(2,:) = Train_Label;
%%% 1. Train_index, 2. Train_Label

%% Create the Training set with randomly sampling 3-D Dataset and its correponding index  
for trial_idx = 1: trial_num

indexes =[];
for i = 1: K_Known
  W_Class_Index = find(Train_Label == i);
  Random_num = randperm(length(W_Class_Index));
  Random_Index = W_Class_Index(Random_num);
  Tr_Index = Random_Index(1:RandSampled_Num(i));
  indexes = [indexes Tr_Index];
end   
indexes = indexes';
train_SL = trainall(:,indexes);
train_samples = img1(train_SL(1,:),:);
train_labels= train_SL(2,:)';

%% Create the Testing set with randomly sampling 3-D Dataset and its correponding index
test_SL = trainall;
test_SL(:,indexes) = [];
test_samples = img1(test_SL(1,:),:);
test_labels = test_SL(2,:)';

%% Generate spectral feature
train_img=zeros(no_rows,no_lines);          
train_img(train_SL(1,:))=train_SL(2,:);

%% Classification based on two feature image
% in_param.other.turning = false;
% 
% in_param.other.CCC = best_param(trial_idx,1); in_param.other.gamma = best_param(trial_idx,2);
% %----------------------------------nu-svm parameter------------------------
% [class_label, out_param] = classify_svm(img,train_img,in_param);

[class_label, out_param, bestng] = classify_svm(HSI,train_img);
best_param = [best_param; bestng];
%% Calculate the error based on predict label and truth label
[OA,kappa,AA,CA] = calcError(test_SL(2,:)-1,class_label(test_SL(1,:))'-1,[1:K_Known]);
overall_OA = [overall_OA;OA]; overall_AA= [overall_AA;AA]; overall_kappa = [overall_kappa;kappa]; overall_CA = [overall_CA, CA];

prediction_map(:,:,trial_idx) = reshape(class_label,no_rows,no_lines);

end
fprintf('OA: %1.4f, AA: %1.4f, kappa: %1.4f', mean(overall_OA), mean(overall_AA), mean(overall_kappa))
'Average accuracy of each class:'
mean(overall_CA,2)
t2=clock;
t=etime(t2,t1);