XX = [];
YY = [];
DataSelected = 2;
Binary = 1;
uplifting = 0;
for i = 1:16    
    RegionSelected = strcat('R',num2str(i));

    fig1 = strcat(RegionSelected,'_original_2019-08-18.tif');
    fig2 = strcat(RegionSelected,'_original_2021-07-23.tif');
    s1 = size(imread(fig1));
    s2 = size(imread(fig2));
    X_2019 = reshape(double(imread(fig1)),s1(1)*s1(2),[]);
    X_2021 = reshape(double(imread(fig2)),s2(1)*s2(2),[]);
    Xu = rgb2lab(X_2021./255)-rgb2lab(X_2019./255);

    clear fig1 fig2 s1 s2 X_2019 X_2021 prompt1

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

    clear fig1 fig2


    % Load and process the labels
    % For binary classification, take abs value of the difference.
    % For multiclass classification, take the difference.
    if Binary == 1
        X = abs(X_2021-X_2019);
        Xu = abs(Xu);
        Y = reshape(double(imread(strcat(RegionSelected,'_original_Binary_change_thr.png'))),s1(1)*s1(2),1);
        Y = Y/255;
    elseif Binary == 0
        X = X_2021-X_2019;
        Y = reshape(double(imread(strcat(RegionSelected,'_original_four_change_thr.png'))),s1(1)*s1(2),1);
    else
        disp('Incorrect prompt input. Please enter one of 1 or 2.')
    end

    if uplifting == 1
        X = [X,Xu];
    end


    HSI = reshape(X,s1(1),s1(2),[]);

    Y2d = reshape(Y,s1(1),s1(2))+1;
    clear prompt

    %
    X1 = X;Y1 = Y;
    XX = [XX;X1];
    YY = [YY;Y1];
end
