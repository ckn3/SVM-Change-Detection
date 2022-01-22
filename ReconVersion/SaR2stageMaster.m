%% Load Dataset and Label

clc
clear
 
profile off;
profile on;

prompt1 = 'Which region? \n Input a number between 1-16 \n';
RegionSelected = strcat('R',num2str(input(prompt1)));

fig1 = strcat(RegionSelected,'_original_2019-08-18.tif');
fig2 = strcat(RegionSelected,'_original_2021-07-23.tif');
s = size(imread(fig1));
X_2019 = reshape(double(imread(fig1)),s(1)*s(2),[]);
X_2021 = reshape(double(imread(fig2)),s(1)*s(2),[]);
X = [X_2019,X_2021];
Xu = [rgb2lab(X_2019./255), rgb2lab(X_2021./255)];

clear fig1 fig2 s1 s2 X_2019 X_2021 prompt1


% Load and process the labels
prompt = 'Select the task \n 1) Binary Classification \n 2) 4-class Classification \n';
choice = input(prompt);
% For binary classification, take abs value of the difference.
% For multiclass classification, take the difference.
if choice == 1
    Y = reshape(double(imread(strcat(RegionSelected,'_original_Binary_change_thr.png'))),s(1)*s(2),1);
    Y = Y/255;
elseif choice == 2
    Y = reshape(double(imread(strcat(RegionSelected,'_original_four_change_thr.png'))),s(1)*s(2),1);
else
    disp('Incorrect prompt input. Please enter one of 1 or 2.')
end

prompt = 'Use Lab colorspace to uplifting the data? \n 1) Yes \n 2) No \n';
choice = input(prompt);

if choice == 1
    X = [X,Xu];
end

HSI = reshape(X,s(1),s(2),[]);
trial_num = 10;
Y2d = reshape(Y,s(1),s(2))+1;
clear prompt4
HSI_ori = HSI;

clear X Xu Y HSI
%%

for pts_per_class = [20,50,100,200]
    HSI = reshape(SA_Recon(HSI_ori,Y2d,pts_per_class,trial_num),s(1),s(2),[]);
    save(strcat('recon',num2str(choice),'_',num2str(pts_per_class)), 'HSI')


    % Find the best parameters of SVM

    K_Known = length(unique(Y2d));
    n=0.01:0.01:0.1;
    dn = (size(HSI,3)+2:-0.5:size(HSI,3)-2);
    dn = dn(dn>1);
    g=1./dn;
    [best_param,idx_all,idx_kappa] = find_best_params(HSI,Y2d,K_Known,trial_num,pts_per_class,n,g);
    n = unique([(best_param(idx_all,1)-0.005):0.0025:(best_param(idx_all,1)+0.005),(best_param(idx_kappa,1)-0.005):0.0025:(best_param(idx_kappa,1)+0.005)]);
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

    save(strcat('recon',num2str(choice),'_',num2str(pts_per_class),'_best'), 'Prediction')
end