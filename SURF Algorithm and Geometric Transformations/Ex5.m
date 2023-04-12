clc;
close all;
clear all;
disp('Ayalla Reuven 314077033')
disp('Lior Avadyayev 206087611')
%%
%1 Mona Liza
%1.1 Inroduction to SURF
%1.1.3
mona_liza=dip_GN_imread("mona_org.jpg");  %grayscale normalized image.
figure;
imshow(mona_liza)
title('Mona Liza')

%1.1.4
tic; mona_corners=detectSURFFeatures(mona_liza); toc;
[mona_features,mona_valid_corners] = extractFeatures(mona_liza,mona_corners);

figure; 
imshow(mona_liza); 
hold on;
plot(mona_valid_corners);
title('Q1.1.4: SURF Features');

%1.1.5
tic; monaROY_corners = detectSURFFeatures(mona_liza,'ROI',[59 5 128 120]); toc;
[monaROI_features, monaROI_valid_points] = extractFeatures(mona_liza, monaROY_corners);  

figure; 
imshow(mona_liza); 
hold on;
rectangle('Position',[59 5 128 120],'EdgeColor','r' );
title('Q1.1.5: Present the selected ’ROI’');

figure; 
imshow(mona_liza); 
hold on;
plot(monaROI_valid_points);
title('Q1.1.5: SURF Features with ROI');

%1.1.6
%NumOctave =1
tic; monaNO1_corners = detectSURFFeatures(mona_liza,'NumOctaves',1); toc;
[monaNO1_features, monaNO1_valid_points] = extractFeatures(mona_liza, monaNO1_corners);   

%NumOctave =2
tic; monaNO2_corners = detectSURFFeatures(mona_liza,'NumOctaves',2); toc;
[monaNO2_features, monaNO2_valid_points] = extractFeatures(mona_liza, monaNO2_corners);   

figure;

subplot(1,2,1);
imshow(mona_liza); 
hold on;
plot(monaNO1_valid_points);
title('Q1.1.6: NumOctaves = 1');

subplot(1,2,2);
imshow(mona_liza); 
hold on;
plot(monaNO2_valid_points);
title('Q1.1.6: NumOctaves = 2');

%1.1.7
%NumOctave =3
tic; monaNSL3_corners = detectSURFFeatures(mona_liza,'NumScaleLevels',3); toc;
[monaNSL3_features, monaNSL3_valid_points] = extractFeatures(mona_liza, monaNSL3_corners);   

%NumOctave =5
tic; monaNSL5_corners = detectSURFFeatures(mona_liza,'NumScaleLevels',5); toc;
[monaNSL5_features, monaNSL5_valid_points] = extractFeatures(mona_liza, monaNSL5_corners);   

figure;
subplot(1,2,1);
imshow(mona_liza); 
hold on;
plot(monaNSL3_valid_points);
title('Q1.1.7: NumScaleLevels = 3');
subplot(1,2,2);
imshow(mona_liza); 
hold on;
plot(monaNSL5_valid_points);
title('Q1.1.7: NumOctaves = 5');
%%
%1.2 Make Mona Straight Again
%1.2.1
mona_crooke_file = fullfile('straight_mona','crooked_mona.jpg');
mona_crooked = imread(mona_crooke_file);

mona_straight_file=fullfile('straight_mona','straight_mona.PNG');
mona_straight = dip_GN_imread(mona_straight_file);

figure;
subplot(1,2,1);
imshow(mona_straight);
title('Q1.2.1: Mona Liza Straight');
subplot(1,2,2);
imshow(mona_crooked);
title('Q1.2.1: Mona Liza Crooked');

%1.2.2
%SURF feature points straight mona
points_s = detectSURFFeatures(mona_straight);
points_c = detectSURFFeatures(mona_crooked);

figure;
subplot(1,2,1)
imshow(mona_straight); 
hold on;
plot(points_s.selectStrongest(10));
title('Q1.2.2: 10 SURF feature');
subplot(1,2,2);
imshow(mona_crooked);
hold on;
plot(points_c.selectStrongest(10));
title('Q1.2.2: 10 SURF feature');

%1.2.3
%% Q1.2.3
[s_features, s_valid_points] = extractFeatures(mona_straight, points_s);        %extract features from the original images.
[c_features, c_valid_points] = extractFeatures(mona_crooked, points_c);         %extract featur from the transformed images

indexPairs = matchFeatures(s_features,c_features);      %Match and display features between the images
s_matched  = s_valid_points(indexPairs(:,1));       
c_matched = c_valid_points(indexPairs(:,2));        
figure;
showMatchedFeatures(mona_straight,mona_crooked,s_matched,c_matched);
title('Q1.2.3: Mona Matched SURF Points');

[tform,inlierIdx] = estimateGeometricTransform(c_matched,s_matched,'similarity');   %estimate the geometric transform matrix, and display the results

mona_recover = imwarp(mona_crooked,tform);     % recover original mona

figure;
subplot(1,2,2);
imshow(mona_recover); 
title('Q1.2.3: Recovered Mona');
subplot(1,2,1);
imshow(mona_straight); 
title('Q1.2.3: Original Mona');
%%
%1.3 Fake Or Real
%1.3.1
monas = dir(['mona' '/*.jpg']);
fake=[]; true=[];

for i=1:length(monas)
    img_file = fullfile('mona',monas(i).name);
    img = dip_GN_imread(img_file);
    
    points = detectSURFFeatures(img);         %SURF feature points mew mona(?)
    [features, valid_points] = extractFeatures(img, points);        %extract features from the new mona(?).
    
    indexPairs = matchFeatures(s_features, features,'MatchThreshold', 5, 'MaxRatio', 0.7, 'Metric', 'SAD');     %Match and display features between the image
    matchedOriginal  = s_valid_points(indexPairs(:,1));
    matchedDistorted = valid_points(indexPairs(:,2));
  
    matc_fetc_num = size(indexPairs,1);
  
    if matc_fetc_num == 0
        disp([img_file(6:end),' is Fake Mona']);
        figure; showMatchedFeatures(mona_straight,img, matchedOriginal,matchedDistorted,'montage');    %display the results
        title('Fake Mona');

    else
        disp([img_file(6:end),' is Real Mona']);
        figure; showMatchedFeatures(mona_straight,img, matchedOriginal,matchedDistorted,'montage');    %display the results
        title('Real Mona');

    end

end
%%
%2
%2.1
figure;
ID2QR('206087611')
hold on
[qr_x,qr_y]=ginput(4);
scatter(qr_x,qr_y,'blue',"filled")
title("206087611_QR")
hold off
fixedpoints= [qr_x(1) qr_y(1);qr_x(2) qr_y(2);qr_x(3) qr_y(3);qr_x(4) qr_y(4)];

%2.2
%-----easy points

easy = imread("easy.jpg");
intermediate = imread("intermediate.jpg");
hard = imread("hard.jpg");
figure;
imshow(easy)
hold on
[easy_x,easy_y]=ginput(4);
scatter(easy_x,easy_y,'blue',"filled")
title('easy')
hold off

%----intermediate points
figure;
imshow(intermediate)
hold on
[inter_x,inter_y]=ginput(4);
scatter(inter_x,inter_y,'blue',"filled")
title('intermediate')
hold off

%2.3
%---hard points
figure;
imshow(hard)
hold on
[hard_x,hard_y]=ginput(4);
scatter(hard_x,hard_y,'blue',"filled")
title('hard')
hold off

%2.4
%----straighten easy image

figure;
st_easy=straighten(easy,[easy_x(1) easy_y(1);easy_x(2) easy_y(2);easy_x(3) easy_y(3);easy_x(4) easy_y(4)],fixedpoints);
st_easy=double(rgb2gray(st_easy));
st_easy=(st_easy-min(st_easy(:)))/(max(st_easy(:))-min(st_easy(:)));

%----straighten intermediate image
figure;
st_inter=straighten(intermediate,[inter_x(1) inter_y(1);inter_x(2) inter_y(2);inter_x(3) inter_y(3);inter_x(4) inter_y(4)],fixedpoints);
st_inter=double(rgb2gray(st_inter));
st_inter=(st_inter-min(st_inter(:)))/(max(st_inter(:))-min(st_inter(:)));

%----straighten hard image
figure;
st_hard=straighten(hard,[hard_x(1) hard_y(1);hard_x(2) hard_y(2);hard_x(3) hard_y(3);hard_x(4) hard_y(4)],fixedpoints);
st_hard=double(rgb2gray(st_hard));
st_hard=(st_hard-min(st_hard(:)))/(max(st_hard(:))-min(st_hard(:))); 

%2.5 -2.6
%-----extract QR code

[easy_mat,id_easy]= QR2ID(st_easy);
[inter_mat,id_inter]= QR2ID(st_inter);
[hard_mat,id_hard] = QR2ID(st_hard);
figure;
imshow(easy_mat)
%%
%3
%3.2
%------ find easy corners
easy_ext= rgb2gray(easy);
[h,w]= size(easy_ext);
easy_ext=easy_ext(50:h-60,50:w-50);
corners = detectHarrisFeatures(easy_ext);
figure;
imshow(easy_ext)
hold on
corners = corners.selectStrongest(25);
plot(corners)
corners=corners.Location;
hold off
easy_corner= find_corners(corners);
figure;
imshow(easy_ext)
hold on
scatter(easy_corner(:,1),easy_corner(:,2),'blue','filled');
hold off

%------ find intermediate corners

figure;
inter_ext= rgb2gray(intermediate);
[h,w]= size(inter_ext);
inter_ext=inter_ext(100:h-200,250:w-100);
corners = detectHarrisFeatures(inter_ext);
figure;
imshow(inter_ext)
hold on
corners = corners.selectStrongest(25);
plot(corners)
corners=corners.Location;
hold off
inter_corner= find_corners(corners);
figure;
imshow(inter_ext)
hold on
scatter(inter_corner(:,1),inter_corner(:,2),'blue','filled');
hold off

%------ find hard corners

hard_ext= rgb2gray(hard);
[h,w]= size(hard_ext);
hard_ext=hard_ext(50:h-60,50:w-400);
corners = detectHarrisFeatures(hard_ext);
figure;
imshow(hard_ext)
hold on
corners = corners.selectStrongest(25);
plot(corners)
corners=corners.Location;
hold off
hard_corner= find_corners(corners);
figure;
imshow(hard_ext)
hold on
scatter(hard_corner(:,1),hard_corner(:,2),'blue','filled');
hold off

%3.4

%----straighten easy image
figure;
st_easy2=straighten(easy_ext,[easy_corner(1,1) easy_corner(1,2);easy_corner(2,1) easy_corner(2,2);easy_corner(3,1) easy_corner(3,2);easy_corner(4,1) easy_corner(4,2)],fixedpoints);


%----straighten intermediate image
figure;
st_inter2=straighten(inter_ext,[inter_corner(1,1) inter_corner(1,2);inter_corner(2,1) inter_corner(2,2);inter_corner(3,1) inter_corner(3,2);inter_corner(4,1) inter_corner(4,2)],fixedpoints);
% st_hard2=straighten(hard,[hard_x(1) hard_y(1);hard_x(2) hard_y(2);hard_x(3) hard_y(3);hard_x(4) hard_y(4)],fixedpoints);

%-----normaliztion

st_easy2=double(st_easy2);
st_easy2=(st_easy2-min(st_easy2(:)))/(max(st_easy2(:))-min(st_easy2(:)));
st_inter2=double(st_inter2);
st_inter2=(st_inter2-min(st_inter2(:)))/(max(st_inter2(:))-min(st_inter2(:))); 

%----extract QR code
[easy_mat2,id_easy2]= QR2ID(double(st_easy2));
[inter_mat2,id_inter2]= QR2ID(st_inter2);
figure;

%%
function[straight_image_3] = straighten(img,movingPoints,fixedpoints)

%Rigid transformation
tform = fitgeotrans(movingPoints,fixedpoints,'NonreflectiveSimilarity');
straight_image_1 = imwarp(img,tform,"OutputView",imref2d([258,258]));
subplot(1,3,1);
imshow(straight_image_1);
title('Rigid');

% Affine transformation
tform = fitgeotrans(movingPoints,fixedpoints,'affine');
straight_image_2 = imwarp(img,tform,"OutputView",imref2d([258,258]));
subplot(1,3,2);
imshow(straight_image_2);
title('Affine');

% Perspective transformation
tform = fitgeotrans(movingPoints,fixedpoints,'projective');
straight_image_3 = imwarp(img,tform,"OutputView",imref2d([258,258]));
subplot(1,3,3);
imshow(straight_image_3);
title('Perspective');
end

function[new_mat,id]= QR2ID(img)
id=[0,0,0,0,0,0,0,0,0];
new_mat=zeros(6,6);
pow=3;
count=0;
id_in=1;
for j=3:41:212   
    for i=3:41:212        
        block= img(i:i+42,j:j+42);
        median1=median(block,"all");
        if(median1>0.5)
            count=count+2^pow;
            new_mat(floor(i/41)+1,floor(j/41)+1)=1;
        end  
        pow=pow-1;
        if (pow==-1)            
            pow=3;
            id(id_in)=count;
            id_in=id_in+1;
            count=0;
        end                
    end
        
end
end

function[corners] = find_corners(harriscorners)
corners=zeros(4,2);
[~, maxIndex] = max(sum(harriscorners,2));
[~, minIndex] =  min(sum(harriscorners,2));
[~, maxdiff] =  max(harriscorners(:,2)-harriscorners(:,1));
[~, mindiff] =  min(harriscorners(:,2)-harriscorners(:,1));

corners(3,:) =  harriscorners(maxdiff,:);
corners(2,:) = harriscorners(minIndex,:);
corners(1,:) =  harriscorners(mindiff,:);
corners(4,:) = harriscorners(maxIndex,:);
end