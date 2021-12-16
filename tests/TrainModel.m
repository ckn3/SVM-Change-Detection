%% Load Dataset and Label

clc
clear
 
profile off;
profile on;

datasets = {'RGB', '6C', '10C'};
Preprocess = {'HistLab', 'Hist'};
Task =   {'Binary','Four'};

prompt1 = 'Which dataset? \n 1) RGB Images \n 2) 6-Channel Images \n 3) 10-Channel Images \n ';
choice1 = input(prompt1);
nbands = sum(1:(choice1+1));

prompt2 = 'Use Lab colorspace to uplifting the data? \n 1) Yes \n 2) No \n';
choice2 = input(prompt2);
uplift = abs(choice2-2);

prompt3 = 'Select the task \n 1) Binary Classification \n 2) 4-class Classification \n';
choice3 = input(prompt3);
if choice3 == 1
    K_Known = 2;
elseif choice3 == 2
    K_Known = 4;
end

dataName = strcat(datasets{choice1},Preprocess{choice2},Task{choice3});
load(strcat(dataName,'.mat'))

clear choice1 choice2 choice3 datasets Preprocess prompt1 prompt2 prompt3 Task
%% Pre-train the nu-SVM using part of labeled data
Xsub = [];
Ysub = [];
for i=1:K_Known
    Xtemp = XX(YY==i-1,:);
    Ytemp = YY(YY==i-1,:);
    Xtemp = Xtemp(round(linspace(1,length(Xtemp),10000)),:);
    Ytemp = Ytemp(round(linspace(1,length(Xtemp),10000)),:);
    Xsub = [Xsub;Xtemp];
    Ysub = [Ysub;Ytemp];
end
HSI=reshape(Xsub,10000*K_Known,1,nbands+uplift*3);
X=Xsub;
Y2d = Ysub+1;

clear XX YY Xsub Ysub Xtemp Ytemp

%%
for pts_per_class = [500,1000,2000,5000,10000]
    n = 0.005:0.005:0.1;
    dn = (nbands+2:-0.5:nbands-2);
    dn = dn(dn>1);
    g=1./dn;
    [best_param,idx_all,idx_kappa] = find_best_params_global(HSI,Y2d,K_Known,10,pts_per_class,n,g);
    n = unique([(best_param(idx_all,1)-0.004):0.001:(best_param(idx_all,1)+0.004),(best_param(idx_kappa,1)-0.004):0.001:(best_param(idx_kappa,1)+0.004)]);
    n = n(n>0);
    g = unique([(best_param(idx_all,2)-0.004):0.001:(best_param(idx_all,2)+0.004),(best_param(idx_kappa,2)-0.004):0.001:(best_param(idx_kappa,2)+0.004)]);
    [best_param,~,~,model] = find_best_params_global(HSI,Y2d,K_Known,10,pts_per_class,n,g);
    save(strcat(dataName,num2str(pts_per_class),'param'), 'best_param','pts_per_class','model')
end
