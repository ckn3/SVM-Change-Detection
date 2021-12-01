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
trial_num = 10;
Y2d = reshape(Y,s1(1),s1(2))+1;
clear choice prompt

%% Find the best parameters of SVM
for pts_per_class = [100,200,300,500]
    K_Known = length(unique(Y));
    n=0.02:0.02:0.2;
    dn = (s1(3)+2:-0.5:s1(3)-2);
    dn = dn(dn>1);
    g=1./dn;
    [best_param,idx_all,idx_kappa] = find_best_params(HSI,Y2d,K_Known,trial_num,pts_per_class,n,g);
    n = unique([(best_param(idx_all,1)-0.015):0.005:(best_param(idx_all,1)+0.015),(best_param(idx_kappa,1)-0.015):0.005:(best_param(idx_kappa,1)+0.015)]);
    n = n(n>0);
    idx1 = find(g == best_param(idx_kappa,2));
    dn1 = (idx1+0.4:-0.1:idx1-0.4);
    dn1 = dn1(dn1>1);
    idx2 = find(g == best_param(idx_all,2));
    dn2 = (idx2+0.4:-0.1:idx2-0.4);
    dn2 = dn2(dn2>1);
    g = unique([1./dn1,1./dn2]);
    best_param = find_best_params(HSI,Y2d,K_Known,trial_num,pts_per_class,n,g);    
    
    par = 0:0.2:1;
    par2 = [0,0.5,1,2];
    Prediction = gridsearch_denoised(HSI,Y2d,K_Known,trial_num,pts_per_class,best_param,par,par2);
    besta1a2 = Prediction.besta1a2;
    disp(besta1a2)
    
    par = (mode(besta1a2(:,1))-0.1):0.05:(mode(besta1a2(:,1))+0.1);
    par = par(par>=0);
    par2 = unique([mode(besta1a2(:,2)),besta1a2(Prediction.idx,2)]);
    Prediction = gridsearch_denoised(HSI,Y2d,K_Known,trial_num,pts_per_class,best_param,par,par2);
    Prediction.best_param = best_param;
    % Produce the best classification results

    save(strcat(num2str(s1(3)),'ch_',num2str(pts_per_class),'_best'), 'Prediction')
end
