%% Load the pretrained model

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

prompt3 = 'Select the task \n 1) Binary Classification \n 2) 4-class Classification \n';
choice3 = input(prompt3);
if choice3 == 1
    K_Known = 2;
elseif choice3 == 2
    K_Known = 4;
end
%
choice4 = [500, 1000, 1500, 2000, 3000];

OAs = zeros(length(choice4),7);
AAs = zeros(length(choice4),7);
kappas = zeros(length(choice4),7);
Jacs = zeros(length(choice4),7);
F1s = zeros(length(choice4),7,K_Known);

for j = 1:length(choice4)
    dataName = strcat(datasets{choice1},Preprocess{choice2},Task{choice3});
    

    for i = 1:7
        RegionSelected = strcat('R',num2str(i));
        load(strcat(RegionSelected,'_',dataName,num2str(choice4(j))), 'Prediction')
        if i == 3
            date1 = 18;date2 = 24;
        elseif i == 4
            date1 = 20;date2 = 23;
        else
            date1 = 18;date2 = 23;
        end
        
        fig1 = strcat(RegionSelected,'_original_2019-08-',num2str(date1),'.tif');
        s1 = size(imread(fig1));

        if choice3 == 1
            Y = reshape(double(imread(strcat(RegionSelected,'_original_Binary_change_thr.png'))),s1(1)*s1(2),1);
            Y = Y/255;
        elseif choice3 == 2
            Y = reshape(double(imread(strcat(RegionSelected,'_original_four_change_thr.png'))),s1(1)*s1(2),1);
        else
            disp('Incorrect prompt input. Please enter one of 1 or 2.')
        end

        %
        Y2d = reshape(Y,s1(1),s1(2))+1;
        [Jac,F1] =JacF1(Y2d, Prediction.pred, K_Known);
        OAs(j,i) = Prediction.OA(Prediction.idx);
        AAs(j,i) = Prediction.AA(Prediction.idx);
        kappas(j,i) = Prediction.kappa(Prediction.idx);
        Jacs(j,i) = mean(Jac);
        F1s(j,i,:) = F1;
    end
    
end
v1 = mean(OAs,2);
v2 = mean(AAs,2);
v3 = mean(kappas,2);
v4 = mean(Jacs,2);
v5 = mean(mean(F1s,3),2);
v = [v1 v2 v3 v4 v5];
% figure
% imagesc([v2,v3])


save(strcat(dataName), 'OAs','AAs','kappas','Jacs','F1s','v')