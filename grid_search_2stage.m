t1=clock;

plot_a1 = []; plot_a2 = []; plot_OA = [];

overall_OA = []; overall_AA = []; overall_kappa = []; overall_CA = [];
best_a1a2 = []; best_param2 = []; 
bestOA=0; bestAA=0; bestkappa=0;

[no_rows,no_lines, no_bands] = size(HSI);
img1=reshape(HSI,[no_rows*no_lines,no_bands]);
prediction_map = zeros(no_rows,no_lines,trial_num);
%% Select the number of training samples for each class
%total label: [46 1428 830 237 483 730 28 478 20 972 2455 593 205 1265 386 93]
% RandSampled_Num = [10 143 83 24 48 73 10 48 10 97 246 59 21 127 39 10]; 
% RandSampled_Num = [20 20 20 20 20 20 14 20 10 20 20 20 20 20 20 20]; 
% RandSampled_Num = [30 30 30 30 30 30 14 30 10 30 30 30 30 30 30 30]; 
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
for a1 = par
    for a2 = par2
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
            [class_label, out_param] = classify_svm_prob(HSI,train_img,in_param); %% output is probability
            
            denoise_predict_tensor = zeros(no_rows, no_lines, no_bands);
            predict_label_prob = reshape(out_param.prob_estimates, no_rows, no_lines, K_Known);
            
            train_map  = zeros(no_rows,no_lines);
            train_map(train_SL(1,:)) = 1;
            
            for i = 1:size(train_SL,2)
                ix = ceil(train_SL(1,i)/no_rows);
                iy = train_SL(1,i)-(ix-1)*no_rows;
                temp = zeros(1,1,K_Known);
                temp(1,1,train_SL(2,i)) = 1;
                predict_label_prob(iy,ix,:) = temp;
            end      %% plug prob in stage 1
            
            denoise_predict_tensor = l2_l1_aniso_l2_less_ADMM_2dir(predict_label_prob,a1,a2,train_map==0,5);
            
            [~,class_label] = max(denoise_predict_tensor,[],3); %% classification rule in stage 2
            
            class_label = reshape(class_label,[],1);
            %% Calculate the error based on predict label and truth label
            [OA,kappa,AA,CA] = calcError(test_SL(2,:)-1,class_label(test_SL(1,:))'-1,[1:K_Known]);
            
            overall_OA = [overall_OA;OA]; 
            overall_AA= [overall_AA;AA]; 
            overall_kappa = [overall_kappa;kappa]; 
            overall_CA = [overall_CA, CA];
            
            prediction_map(:,:,trial_idx) = reshape(class_label,no_rows,no_lines);
        end
        [~,idx] = max(overall_OA+overall_AA+overall_kappa);
        fprintf('OA: %1.4f, AA: %1.4f, kappa: %1.4f', overall_OA(idx), overall_AA(idx), overall_kappa(idx))
        disp('Best accuracy of each class:')
        overall_CA(:,idx)
        if (max(overall_OA+overall_AA+overall_kappa) >= bestOA+bestAA+bestkappa)
            [~,idx] = max(overall_OA+overall_AA+overall_kappa);
            bestOA = overall_OA(idx);
            bestAA = overall_AA(idx);
            bestkappa = overall_kappa(idx);
            bestCA = overall_CA(:,idx);
            best_a1 = a1;
            best_a2 = a2;
        end
        
    end
end
best_a1a2 = [best_a1 best_a2];
fprintf('bestOA: %1.4f, bestAA: %1.4f, bestkappa: %1.4f', bestOA, bestAA, bestkappa)
disp('Average accuracy of each class:')
disp(bestCA)

% mean(overall_CA,2) %% each row is averaged
t2=clock;
t=etime(t2,t1);