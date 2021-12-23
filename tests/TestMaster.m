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
uplift = abs(choice2-2);

prompt3 = 'Select the task \n 1) Binary Classification \n 2) 4-class Classification \n';
choice3 = input(prompt3);
if choice3 == 1
    K_Known = 2;
elseif choice3 == 2
    K_Known = 4;
end

% %
trial_num = 20;

%

prompt4 = 'Enter Points per class for Training (1000, 2000, 3000) \n';
choice4 = input(prompt4);

for j = 1:length(choice4)
    dataName = strcat(datasets{choice1},Preprocess{choice2},Task{choice3});
    load(strcat(dataName,num2str(choice4(j)),'param'))

    clear prompt1 prompt2 prompt3 prompt4
    clear overall_AA overall_CA overall_kappa overall_OA prediction_map

    for i = 1:7
        RegionSelected = strcat('R',num2str(i));

        if i == 3
            date1 = 18;date2 = 24;
        elseif i == 4
            date1 = 20;date2 = 23;
        else
            date1 = 18;date2 = 23;
        end


        fig1 = strcat(RegionSelected,'_original_2019-08-',num2str(date1),'.tif');
        fig2 = strcat(RegionSelected,'_original_2021-07-',num2str(date2),'.tif');
        s1 = size(imread(fig1));
        s2 = size(imread(fig2));
        X_2019 = reshape(double(imread(fig1)),s1(1)*s1(2),[]);
        X_2021 = reshape(double(imread(fig2)),s2(1)*s2(2),[]);
        Xu = rgb2lab(X_2021./255)-rgb2lab(X_2019./255);

        clear fig1 fig2 s1 s2 X_2019 X_2021

        if choice1 == 1
            fig1 = strcat(RegionSelected,'_original_2019-08-',num2str(date1),'.tif');
            fig2 = strcat(RegionSelected,'_original_2021-07-',num2str(date2),'.tif');
            s1 = size(imread(fig1));
            s2 = size(imread(fig2));
            X_2019 = reshape(double(imread(fig1)),s1(1)*s1(2),[]);
            X_2021 = reshape(double(imread(fig2)),s2(1)*s2(2),[]);

        elseif choice1 ==2
            fig1 = load(strcat(RegionSelected,'_original_2019-08-',num2str(date1),'_ch6.mat'));
            fig2 = load(strcat(RegionSelected,'_original_2021-07-',num2str(date2),'_ch6.mat'));
            s1 = size(fig1.new);
            s2 = size(fig2.new);
            X_2019 = reshape(double(fig1.new),s1(1)*s1(2),[]);
            X_2021 = reshape(double(fig2.new),s2(1)*s2(2),[]);

        elseif choice1 == 3 
            fig1 = load(strcat(RegionSelected,'_original_2019-08-',num2str(date1),'_ch10.mat'));
            fig2 = load(strcat(RegionSelected,'_original_2021-07-',num2str(date2),'_ch10.mat'));
            s1 = size(fig1.new);
            s2 = size(fig2.new);
            X_2019 = reshape(double(fig1.new),s1(1)*s1(2),[]);
            X_2021 = reshape(double(fig2.new),s2(1)*s2(2),[]);

        else
            disp('Incorrect prompt input. Please enter one of [1:3].')
        end

        clear fig1 fig2

        if choice3 == 1
            X = abs(X_2021-X_2019);
            Xu = abs(Xu);
            Y = reshape(double(imread(strcat(RegionSelected,'_original_Binary_change_thr.png'))),s1(1)*s1(2),1);
            Y = Y/255;
        elseif choice3 == 2
            X = X_2021-X_2019;
            Y = reshape(double(imread(strcat(RegionSelected,'_original_four_change_thr.png'))),s1(1)*s1(2),1);
        else
            disp('Incorrect prompt input. Please enter one of 1 or 2.')
        end

        if uplift == 1
            X = [X,Xu];
        end

        % Set the denoising parameters based on experiments on the 16 regions.
        
        if nbands == 3 && uplift == 0 % RGB Hist Binary/Four
            par = 0;
            par2 = 2;
        elseif nbands == 3 && uplift == 1 % RGB HistLab Binary/Four
            par = 0.05;
            par2 = 2;
        elseif nbands == 6 && uplift == 0 && choice3 == 1 % 6ch Hist Binary
            par = 0;
            par2 = 1;
        elseif nbands == 6 && uplift == 1 && choice3 == 1 % 6ch HistLab Binary
            par = 0.05;
            par2 = 1;
        elseif nbands == 6 && uplift == 0 && choice3 == 2 % 6ch Hist Four
            par = 0.1;
            par2 = 0.5;
        elseif nbands == 6 && uplift == 1 && choice3 == 2 % 6ch HistLab Four
            par = 0.15;
            par2 = 0;
        elseif nbands == 10 && choice3 == 1 % 10ch Hist/HistLab Binary
            par = 0.1;
            par2 = 0.5;
        elseif nbands == 10 && choice3 == 2 % 10ch Hist/HistLab Four
            par = 0.15;
            par2 = 0;
        end


        clear prompt X_2019 X_2021

        %
        HSI = reshape(X,s1(1),s1(2),[]);
        Y2d = reshape(Y,s1(1),s1(2))+1;
        pts_per_class = 1;
        Prediction = gridsearch_train_model(HSI,Y2d,K_Known,trial_num,pts_per_class,best_param,par,par2,model);
        save(strcat(RegionSelected,'_',dataName,num2str(choice4(j))), 'Prediction','choice4')
    end
end