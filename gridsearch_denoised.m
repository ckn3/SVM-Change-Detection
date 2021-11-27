function Prediction = gridsearch_denoised(HSI,Y2d,K_Known,trial_num,pts_per_class,best_param,par,par2)
besta1a2=[];
[no_rows,no_lines, no_bands] = size(HSI);
img1=reshape(HSI,[no_rows*no_lines,no_bands]);
prediction_map = zeros(no_rows,no_lines,trial_num);
%% Select the number of training samples for each class
RandSampled_Num =  repmat(pts_per_class,1,K_Known);

Nonzero_map = zeros(no_rows,no_lines);
Nonzero_index =  find(Y2d ~= 0);      %% find labeled pixels(colored pixels)
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
rng('default')
rng(1) %random seed
overall_OA = []; overall_AA = []; overall_kappa = []; overall_CA = [];
train_SL = []; 
indexes =[];

for trial_idx = 1: trial_num

    indexes =[];
    for i = 1: K_Known
        W_Class_Index = find(Train_Label == i); % based on teainall
        Random_num = randperm(length(W_Class_Index));
        Random_Index = W_Class_Index(Random_num);
        Tr_Index = Random_Index(1:RandSampled_Num(i));
        indexes = [indexes Tr_Index];
    end
    indexes = indexes';
    train_SL = trainall(:,indexes); % based on teainall
    train_samples = img1(train_SL(1,:),:); % vectors of training datas
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
    in_param.other.turning = false; %true for fivefold cross-validation

    in_param.other.CCC = best_param(trial_idx,1); in_param.other.gamma = best_param(trial_idx,2);
    [~, out_param] = classify_svm_prob(HSI,train_img,in_param); %% output is probability
    
    train_map  = zeros(no_rows,no_lines);
    train_map(train_SL(1,:)) = 1;

    predict_label_prob = reshape(out_param.prob_estimates, no_rows, no_lines, K_Known);

    for i = 1:size(train_SL,2)
        ix = ceil(train_SL(1,i)/no_rows);
        iy = train_SL(1,i)-(ix-1)*no_rows;
        temp = zeros(1,1,K_Known);
        temp(1,1,train_SL(2,i)) = 1;
        predict_label_prob(iy,ix,:) = temp;
    end      %% plug prob in stage 1
    
    [class_label_denoised, best_denoise_param] = denoisingL1L2(predict_label_prob, par,par2,train_map, test_SL, K_Known);
    %% Calculate the error based on predict label and truth label
    [OA,kappa,AA,CA] = calcError(test_SL(2,:)-1,class_label_denoised(test_SL(1,:))'-1,1:K_Known);

    overall_OA = [overall_OA;OA]; 
    overall_AA= [overall_AA;AA]; 
    overall_kappa = [overall_kappa;kappa]; 
    overall_CA = [overall_CA, CA];
    besta1a2(trial_idx,:) = best_denoise_param;
    prediction_map(:,:,trial_idx) = reshape(class_label_denoised,no_rows,no_lines);
end
[~,idx] = max(overall_OA+overall_AA+overall_kappa);
fprintf('OA: %1.4f, AA: %1.4f, kappa: %1.4f', overall_OA(idx), overall_AA(idx), overall_kappa(idx))
disp('Best accuracy of each class:')
overall_CA(:,idx)
Prediction.map = prediction_map;
Prediction.pred = prediction_map(:,:,idx);
Prediction.OA = overall_OA(idx);
Prediction.AA = overall_AA(idx);
Prediction.kappa = overall_kappa(idx);
Prediction.CA = overall_CA(:,idx);
Prediction.idx = idx;
Prediction.besta1a2=besta1a2;
end