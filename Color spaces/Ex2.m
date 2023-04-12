clc;
clear all;
clearvars;
disp('Ayalla Reuven 314077033')
disp('Lior Avadyayev 206087611')

%1.1
city=double(imread('colorful-cities.jpg'));   %read and convert image to "double"
city=(city-min(city(:)))./(max(city(:))-min(city(:)));  %normalize image

%1.2
figure(1);                                                    
imshow(city);                                                           
title('Q1.2: Color City');

%1.3
red_city = city(:,:,1);
green_city = city(:,:,2);
blue_city = city(:,:,3);

figure;  
subplot(3,1,1);
imshow(red_city);
title('Q1.3: Color City - Red channel');
subplot(3,1,2);
imshow(green_city);
title('Q1.3: Color City - Green channel');
subplot(3,1,3);
imshow(blue_city);
title('Q1.3: Color City - Blue channel');
colormap;

%1.4
our_gray_city=dip_rgb2gray(city);

%1.5
matlab_gray_city=rgb2gray(city);
gray_city_mse=abs(our_gray_city-matlab_gray_city);  %measure to show that the images are similar
e=max(gray_city_mse,[],'all');

figure;
subplot(1,2,1);
imshow(our_gray_city);
title("Q1.5:  grayscale Color City by our function")
subplot(1,2,2);
imshow(matlab_gray_city);
title("Q1.5:  grayscale Color City by matlab function")

%1.6
%manipulation1: linear function a*I +b
F = @(I,a,b) a*I+b;
new_city1= city;
new_red = F(red_city,-1,1); %linear function of the red channel
new_city1(:,:,1) = new_red;
new_city1(:,:,2) = green_city;
new_city1(:,:,3) = blue_city;

%manipulation2: switch between the channels
new_city2= city;
new_city2(:,:,1) = blue_city; % 3 to 1
new_city2(:,:,2) = red_city; % 1 to 2
new_city2(:,:,3) = green_city; % 2 to 3

%manipulation3: 0.1x reduction of the green channel
new_city3=city;
new_green = 0.1*green_city; 
new_city3(:,:,1) = red_city;
new_city3(:,:,2) = new_green;
new_city3(:,:,3) = blue_city;

figure; 
subplot(2,2,1)
imshow(city);
title('Q1.6: original city');
subplot(2,2,2);
imshow(new_city1);
title('Q1.6: city after manipulations by linear function');
subplot(2,2,3);
imshow(new_city2);
title('Q1.6: city after manipulations by swtiched channel');
subplot(2,2,4);
imshow(new_city3);
title('Q1.6: city after 0.1x reduction of the green channel');

%2.2
black_city=min(min(1-red_city,1-green_city),1-blue_city);
cyan_city=(1-red_city-black_city)./(1-black_city);
magenta_city=(1-green_city-black_city)./(1-black_city);
yellow_city=(1-blue_city-black_city)./(1-black_city);

figure;
subplot(2,2,1);
imshow(black_city);
title('Q2.2: CYMK- Black channel');
subplot(2,2,2);
imshow(cyan_city);
title('Q2.2: CYMK- Cyan channel');
subplot(2,2,3);
imshow(magenta_city);
title('Q2.2: CYMK- Magenta channel');
subplot(2,2,4);
imshow(yellow_city);
title('Q2.2: CYMK- Yellow channel');

%2.3
displayCYMK(cyan_city,yellow_city,magenta_city,black_city);

%2.4
%manipulation1: linear function a*I +b
F = @(I,a,b) a*I+b;
new_cyan = F(cyan_city,-1,1); %linear function of the cyan channel

%manipulation2: switch between the channels

%manipulation3: 0.1x reduction of the maenta channel
new_magneta= 0.1*magenta_city;

figure; 
subplot(2,2,1)
imshowCYMK(cyan_city,yellow_city,magenta_city,black_city);
title('Q2.4: original city');
subplot(2,2,2);
imshowCYMK(new_cyan,yellow_city,magenta_city,black_city);
title('Q2.4: city after manipulations by linear function');
subplot(2,2,3);
imshowCYMK(black_city,magenta_city,yellow_city,cyan_city);
title('Q2.4: city after manipulations by swtiched channel');
subplot(2,2,4);
imshowCYMK(cyan_city,yellow_city,new_magneta,black_city);
title('Q2.4: city after 0.1x reduction of the magenta');

%3.3-3.4
[hue_city,saturation_city,value_city]=dip_rgb2hsv(red_city,green_city,blue_city);
[hue_city_matlab,saturation_city_matlab,value_city_matlab]=rgb2hsv(red_city,green_city,blue_city);

hue_city_mse=abs(hue_city-hue_city_matlab);  %measure to show that the hue are similar
e1=max(hue_city_mse,[],'all');
value_city_mse=abs(value_city-value_city_matlab);  %measure to show that the value are similar
e2=max(value_city_mse,[],'all');
saturation_city_mse=abs(saturation_city-saturation_city_matlab);  %measure to show that the saturation are similar
e3=max(saturation_city_mse,[],'all');


figure; 
subplot(3,2,1);
imshow(hue_city);
title('Q3.4: city - hue channel');
subplot(3,2,3);
imshow(saturation_city);
title('Q3.4: city - saturation channel');
subplot(3,2,5);
imshow(value_city);
title('Q3.4: city - value channel');
subplot(3,2,2);
imshow(hue_city_matlab);
title('Q3.4: city - hue channel by matlab');
subplot(3,2,4);
imshow(saturation_city_matlab);
title('Q3.4: city - saturation channel by matlab');
subplot(3,2,6);
imshow(value_city_matlab);
title('Q3.4: city - value channel by matlab');

%3.5
%manipulation1: linear function a*I +b
F = @(I,a,b) a*I+b;
new_value = F(value_city,-1,1); %linear function of the value channel

%manipulation2: switch between the channels

%manipulation3: 0.1x reduction of the saturation channel
new_saturation=0.1*saturation_city;  


figure; 
subplot(2,2,1)
imshowHSV(hue_city,saturation_city,value_city);
title('Q3.5: original city');
subplot(2,2,2);
imshowHSV(hue_city,saturation_city,new_value);
title('Q3.5: city after manipulations by linear function');
subplot(2,2,3);
imshowHSV(hue_city,value_city,saturation_city);
title('Q3.5: city after manipulations by swtiched channel');
subplot(2,2,4);
imshowHSV(hue_city,new_saturation,value_city);
title('Q3.5: city after 0.1x reduction of the saturation channel');
colormap('hsv')

%3.6
[hue_newcity2,saturation_newcity2,value_newcity2]=dip_rgb2hsv(green_city,blue_city,red_city);

figure; 
subplot(3,2,1);
imshow(hue_city);
title('Q3.6: city - hue channel');
subplot(3,2,3);
imshow(saturation_city);
title('Q3.6: city - saturation channel');
subplot(3,2,5);
imshow(value_city);
title('Q3.6: city - value channel');
subplot(3,2,2);
imshow(hue_newcity2);
title('Q3.6: city - hue channel after switching RGB channels');
subplot(3,2,4);
imshow(saturation_newcity2);
title('Q3.6: city - saturation channel after switching RGB channels');
subplot(3,2,6);
imshow(value_newcity2);
title('city - value channel after switching RGB channels');

%4.2

LAB_city = rgb2lab(city);

LAB_L_city = LAB_city(:,:,1);
LAB_A_city = LAB_city(:,:,2);
LAB_B_city = LAB_city(:,:,3);

figure;  
subplot(1,3,1);
imshow(LAB_L_city);
title('Q4.2: Color City - L channel');
subplot(1,3,2);
imshow(LAB_A_city);
title('Q4.2: Color City - A channel');
subplot(1,3,3);
imshow(LAB_B_city);
title('Q4.2: Color City - B channel');
colormap;


%4.3
%manipulation1: linear function a*I +b
F = @(I,a,b) a*I+b;
new_L = F(LAB_L_city ,1,-40); %linear function of the red channel


%manipulation2: switch between the channels - done in the imshowLab inputs



%manipulation3: 0.1x reduction of the saturation channel
new_A=LAB_A_city*-1;  

figure; 
subplot(2,2,1)
imshowLab(LAB_L_city,LAB_A_city,LAB_B_city);
title('Q4.2: original city');
subplot(2,2,2);
imshowLab(new_L,LAB_A_city,LAB_B_city);
title('Q4.2: city after manipulations of L channel by linear function');
subplot(2,2,3);
imshowLab(LAB_B_city,LAB_L_city,LAB_A_city);
title('Q4.2: city after manipulations by swtiched channel');
subplot(2,2,4);
imshowLab(LAB_L_city,new_A,LAB_B_city);
title('Q4.2:city after (-1) multiplication of the A channel');



%4.4

figure; 
subplot(3,2,1);
imshow(hue_city);
title('Q4.4: city - hue channel');
subplot(3,2,3);
imshow(saturation_city);
title('Q4.4: city - saturation channel');
subplot(3,2,5);
imshow(value_city);
title('Q4.4: city - value channel');
subplot(3,2,2);
imshow(LAB_L_city);
title('Q4.4: city - L channel ');
subplot(3,2,4);
imshow(LAB_A_city);
title('Q4.4: city - A channel ');
subplot(3,2,6);
imshow(LAB_B_city);
title('Q4.4: city - B channel ');

%5
cap1=double(imread("cap1.png"));
cap2=double(imread("cap2.png"));
cap3=double(imread("cap3.png"));
cap1=(cap1-min(cap1(:)))./(max(cap1(:))-min(cap1(:))); 
cap2=(cap2-min(cap2(:)))./(max(cap2(:))-min(cap2(:))); 
cap3=(cap3-min(cap3(:)))./(max(cap3(:))-min(cap3(:))); 

% finding cap in every image using find cap function written bellow
cap1_found=find_cap(cap1,200);
cap2_found=find_cap(cap2,210);
cap3_found=find_cap(cap3,220);



function [Gray]=dip_rgb2gray(img)   %Q1.4
R=img(:,:,1);
G=img(:,:,2);
B=img(:,:,3);
Gray=0.2989*R+0.5870*G+0.1140*B;
end

function [hue,saturation,value] = dip_rgb2hsv(red,green,blue)
x_max=max(max(red,green),blue);     %value(V)
x_min=min(min(red,green),blue);     %V-C
C = x_max-x_min;    %2(V-L)
%L=(x_min+x_max)./2;     %V-C/2
hue=zeros(size(red));
saturation=zeros(size(red));
for i=1:size(red,1)
    for j=1:size(red,2)
        if C(i,j)==0
            hue(i,j)=0;
        elseif x_max(i,j)==red(i,j)
            hue(i,j)=(1/6)*mod((green(i,j)-blue(i,j))/C(i,j),6);
        elseif x_max(i,j)==green(i,j)
            hue(i,j)=(1/6)*(2+(blue(i,j)-red(i,j))/C(i,j));
        elseif x_max(i,j)==blue(i,j)
            hue(i,j)=(1/6)*(4+(red(i,j)-green(i,j))/C(i,j));
        end
        
        if x_max(i,j)==0
            saturation(i,j)=0;
        else 
            saturation(i,j)=C(i,j)/x_max(i,j);
        end
        
%         if (L==zeros(size(red)))||L==ones(size(red))
%             s_l=zeros(size(red));
%         else
%             s_l=(x_max-L)./(min(L,ones(size(red))-L));
%         end
    end
end
value=x_max;
end


function [img_median] = median_filter(img,k)  %median filter
[hei,wid]= size(img);
img_median=zeros(hei,wid);
cut = floor(k/2);
mat=zeros(hei+k-1,wid+k-1);
mat(cut+1:hei+cut,cut+1:wid+cut)= img;
for i=1:hei-cut
    for j=1:wid-cut
        mat1=mat(i:i+k-1,j:j+k-1);
        img_median(i,j)= median(mat1,"all");
        
    end
end
end

function newIm = draw_circle(img,cap)
    
    [r,c] = size(img); % get size
    sum_rows = img*ones(c,1);  % sum all rows
    sum_cols = img'*ones(r,1); % sum all columns
    x0 = find(sum_cols~=0, 1, 'first'); % find x value of left
    x1 = find(sum_cols~=0, 1, 'last');  % find x value of right
    y0 = find(sum_rows~=0, 1, 'first'); % find y value of upp
    y1 = find(sum_rows~=0, 1, 'last');  % find y value of down
    % find the center of the cap
    x = floor((x0+x1)/2);   
    y = floor((y0+y1)/2);
    % find radius of circle
    r = sqrt(((x0-x1)/2)^2+((y0-y1)/2)^2)+5;
    % circle cup in image
    newIm = insertShape(cap,'circle',[x,y,r],'LineWidth',5);
end


function image_cap = find_cap(img,k) %circling the cap in image
%-----rgb part
figure(k);
imshow(img)
img_red=img(:,:,1);
img_green=img(:,:,2);
img_blue=img(:,:,3);
% setting black images
black_img_rgb=zeros(size(img_blue)); 
black_img_hsv=zeros(size(img_blue));
%RGB color filtering
black_img_rgb(and(and(and(img_red<0.25,img_blue>0.08),img_blue<0.4),img_green<0.1))=1; 

%setting the output RGB image
img_red(black_img_rgb<1)=0;
img_green(black_img_rgb<1)=0;
img_blue(black_img_rgb<1)=0;

%HSV color filtering
[black_hue_city,black_saturation_city,black_value_city]=rgb2hsv(img_red,img_green,img_blue);
black_img_hsv(and(and(and(black_hue_city>3/5,black_hue_city<3/4),black_value_city>0.10),black_saturation_city>0.4))=1;
figure(k+1);
imshow(black_img_rgb)

title("after RGB filtering")
figure(k+2);
imshow(black_img_hsv)
title("after HSV filtering")
black_img_hsv_med=median_filter(black_img_hsv,13);  %median filter
image_cap=draw_circle(black_img_hsv_med,img);
figure(k+4);
imshow(image_cap)
title("with circle")
end


