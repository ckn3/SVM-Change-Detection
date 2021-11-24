%% Load Dataset and Label

clc
clear
 
profile off;
profile on;

prompt1 = 'Which region? \n Input a number between 1-16 \n';
RegionSelected = strcat('R',num2str(input(prompt1)));

fig1 = strcat(RegionSelected,'_original_2019-08-18.tif');
fig2 = strcat(RegionSelected,'_original_2021-07-23.tif');
s1 = size(imread(fig1));
s2 = size(imread(fig2));
X_2019 = reshape(double(imread(fig1)),s1(1)*s1(2),[]);
X_2021 = reshape(double(imread(fig2)),s2(1)*s2(2),[]);
Xu = rgb2lab(X_2021./255)-rgb2lab(X_2019./255);

clear fig1 fig2 s1 s2 X_2019 X_2021 prompt1

prompt2 = 'Which dataset? \n 1) RGB Images \n 2) 6-Channel Images \n 3) 10-Channel Images \n ';
DataSelected = input(prompt2);

if DataSelected == 1
    fig1 = strcat(RegionSelected,'_original_2019-08-18.tif');
    fig2 = strcat(RegionSelected,'_original_2021-07-23.tif');
    s1 = size(imread(fig1));
    s2 = size(imread(fig2));
    X_2019 = reshape(double(imread(fig1)),s1(1)*s1(2),[]);
    X_2021 = reshape(double(imread(fig2)),s2(1)*s2(2),[]);
    
elseif DataSelected ==2
    fig1 = load(strcat(RegionSelected,'_original_2019-08-18_ch6.mat'));
    fig2 = load(strcat(RegionSelected,'_original_2021-07-23_ch6.mat'));
    s1 = size(fig1.new);
    s2 = size(fig2.new);
    X_2019 = reshape(double(fig1.new),s1(1)*s1(2),[]);
    X_2021 = reshape(double(fig2.new),s2(1)*s2(2),[]);
    
elseif DataSelected == 3 
    fig1 = load(strcat(RegionSelected,'_original_2019-08-18_ch10.mat'));
    fig2 = load(strcat(RegionSelected,'_original_2021-07-23_ch10.mat'));
    s1 = size(fig1.new);
    s2 = size(fig2.new);
    X_2019 = reshape(double(fig1.new),s1(1)*s1(2),[]);
    X_2021 = reshape(double(fig2.new),s2(1)*s2(2),[]);
    

else
    disp('Incorrect prompt input. Please enter one of [1:3].')
end

clear fig1 fig2 DataSelected prompt2


% Load and process the labels
prompt = 'Select the task \n 1) Binary Classification \n 2) 4-class Classification \n';
choice = input(prompt);
% For binary classification, take abs value of the difference.
% For multiclass classification, take the difference.
if choice == 1
    X = abs(X_2021-X_2019);
    Xu = abs(Xu);
    Y = reshape(double(imread(strcat(RegionSelected,'_original_Binary_change_thr.png'))),s1(1)*s1(2),1);
    Y = Y/255;
elseif choice == 2
    X = X_2021-X_2019;
    Y = reshape(double(imread(strcat(RegionSelected,'_original_four_change_thr.png'))),s1(1)*s1(2),1);
else
    disp('Incorrect prompt input. Please enter one of 1 or 2.')
end

prompt = 'Use Lab colorspace to uplifting the data? \n 1) Yes \n 2) No \n';
choice = input(prompt);

if choice == 1
    X = [X,Xu];
end


HSI = reshape(X,s1(1),s1(2),[]);

Y2d = reshape(Y,s1(1),s1(2))+1;
clear choice prompt
%% Find the best parameters of SVM
for pts_per_class = [100,200,300,500]
    K_Known = length(unique(Y));
    find_best_params
    par = 0:0.2:1;
    par2 = [0:0.5:1,2];
    grid_search_2stage

    disp(best_a1a2)
    % A finer grid,and considers the case that no denoising is applied.
    par = (best_a1-0.1):0.05:(best_a1+0.1);
    par = par(par>=0);
    par2 = best_a2;
    grid_search_2stage
    % Produce the best classification results
    par = best_a1a2(1);
    par2 = best_a1a2(2);
    grid_search_2stage
    
    clc
    [~,idx]=max(overall_OA+overall_AA+overall_kappa);
    pred = prediction_map(:,:,idx);
    disp(['OA: ',num2str(overall_OA(idx))])
    disp(['AA: ',num2str(overall_AA(idx))])
    disp(['kappa: ',num2str(overall_kappa(idx))])
    disp(['class acc: ',num2str(reshape(overall_CA(:,idx),1,K_Known))])

    save(strcat(num2str(s1(3)),'ch_',num2str(pts_per_class),'_best'), 'pred', 'best_a1a2','best_param', 'idx', 'overall_AA','overall_CA','overall_kappa','overall_OA', 'prediction_map','pts_per_class')
end
