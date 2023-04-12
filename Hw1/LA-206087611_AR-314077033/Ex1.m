clear all
clearvars
disp('Ayalla Reuven 314077033')
disp('Lior Avdayev 206087611')



%% 
%%%1.1%%%
picasso=imread('picasso.jpg');  
picassogray=double(rgb2gray(picasso)); 

figure;
imagesc(picasso)
title('Picasso Original Img')

figure;
imagesc(picassogray)
colormap("gray")
colorbar
title('Picasso Gray Img')

normgraypicasso=dip_GN_imread('picasso.jpg');
figure;
imagesc(normgraypicasso)
colormap("gray")
colorbar
title('Normalizes Gray Img')
%% 

%%%1.2%%%
figure;
subplot(2,1,1)
my_hist=dip_histogram(picassogray,256);
title('Histogram by our function; 256 bins');
subplot(2,1,2)
imhist(rgb2gray(picasso));
title('Histogram by matlab function; 256 bins')

comp_res= my_hist-imhist(rgb2gray(picasso));    %Compare our result to MATLAB imhist() function

figure;
subplot(2,1,1)
my_hist_4=dip_histogram(rgb2gray(picasso),4);
title('Histogram by our function; 4 bins');
subplot(2,1,2)
imhist(rgb2gray(picasso),4);
title('Histogram by matlab function; 4 bins')

figure;
subplot(2,1,1)
my_hist_32=dip_histogram(picassogray,32);
title('Histogram by our function; 32 bins');
subplot(2,1,2)
imhist(rgb2gray(picasso),32);
title('Histogram by matlab function; 32 bins')

figure;
subplot(2,1,1)
my_hist_128=dip_histogram(picassogray,128);
title('Histogram by our function; 128 bins');
subplot(2,1,2)
imhist(rgb2gray(picasso),128);
title('Histogram by matlab function; 128 bins')
%% 

%%%1.3%%%
% increased and decreased brightness
im = normgraypicasso;  
brightim1 = adjust_brightness(normgraypicasso,'add',0.4); 
brightim2 = adjust_brightness(normgraypicasso,'add',0.6);
brightim3 = adjust_brightness(normgraypicasso,'mul',1.4);
brightim4=adjust_brightness(normgraypicasso,'mul',1.6);

figure;
subplot(1,5,1);
imshow(im)
title('orignal image')
subplot(1,5,2);
imshow(brightim1)
title('Image brightness + 0.4')
subplot(1,5,3);
imshow(brightim2)
title('Image brightness + 0.6')
subplot(1,5,4);
imshow(brightim3)
title('Image brightness * 1.4')
subplot(1,5,5);
imshow(brightim4)
title('Image brightness * 1.6')
%% 

%%%1.4%%%
%linear mapping
contimg1=adjust_contrast(normgraypicasso,0.45,0.9);
figure;
subplot(3,2,1);
imshow(contimg1)
title('contrast range: 0.45 - 0.9')
colorbar
subplot(3,2,2);
dip_histogram(256*contimg1,256);
title('The histogram')

contimg2=adjust_contrast(normgraypicasso,0.4,0.5);
subplot(3,2,3);
imshow(contimg2)
title('contrast range: 0.4 - 0.5')
colorbar
subplot(3,2,4);
dip_histogram(256*contimg2,256);
title('The histogram')

contimg3=adjust_contrast(normgraypicasso,1,0);
subplot(3,2,5);
imshow(contimg3)
title('contrast range: 1 - 0')
colorbar
subplot(3,2,6);
dip_histogram(256*contimg3,256);
title('The histogram')

%non linear mapping
nonlinimg1 = normgraypicasso;
nonlinimg1(normgraypicasso>0.9)=0.9;
nonlinimg1(normgraypicasso<0.45)=0.45;
figure;
subplot(1,2,1);
imshow(nonlinimg1)
title('Non-linear contrast image with [0.45,0.9] range');
subplot(1,2,2);
dip_histogram(256*nonlinimg1,256);
title('Histogram of non-linear contrast image with [0.45,0.9] range');

nonlinimg2 = normgraypicasso;
nonlinimg2(normgraypicasso>08)=0.8;
nonlinimg2(normgraypicasso<0.2)=0.2;
figure;
subplot(1,2,1);
imshow(nonlinimg2)
title('Non-linear contrast image with [0.2,0.8] range');
subplot(1,2,2);
dip_histogram(256*nonlinimg2,256);
title('Histogram of non-linear contrast image with [0.45,0.9] range');
%% 

%%%1.5%%%
figure;
subplot(1,4,1)
quanpicasso1 = floor(picassogray/(2^(8-1)));  
normquanpicasso1 = (quanpicasso1-min(quanpicasso1(:)))/(max(quanpicasso1(:))-min(quanpicasso1(:)));  
imshow(normquanpicasso1);
title('Quantization by 1 bits')

quanpicasso2 = floor(picassogray/(2^(8-2)));  
normquanpicasso2 = (quanpicasso2-min(quanpicasso2(:)))/(max(quanpicasso2(:))-min(quanpicasso2(:)));
subplot(1,4,2)
imshow(normquanpicasso2);
title('Quantization by 2 bits')

quanpicasso4 = floor(picassogray/(2^(8-4)));  
normquanpicasso4 = (quanpicasso4-min(quanpicasso4(:)))/(max(quanpicasso4(:))-min(quanpicasso4(:)));
subplot(1,4,3)
imshow(normquanpicasso4);
title('Quantization by 4 bits')

quanpicasso6 = floor(picassogray/(2^(8-6))); 
normquanpicasso6 = (quanpicasso6-min(quanpicasso6(:)))/(max(quanpicasso6(:))-min(quanpicasso6(:)));
subplot(1,4,4)
imshow(normquanpicasso6);
title('Quantization by 6 bits')
%% 

%%%1.6%%%
dog = imread('dog.jpg');
graydog = double(rgb2gray(dog)); 
normgraydog = dip_GN_imread('dog.jpg');
hisdog = histeq(normgraydog);

figure;
subplot(2,2,1);
imshow(normgraydog);
colorbar
title('Normalized image')
subplot(2,2,2);
dip_histogram(256*normgraydog,256);
title('The histogram of normalized image')
subplot(2,2,3);
histeq(normgraydog)
title('histogram equalization image')
colorbar
subplot(2,2,4);
dip_histogram(256*hisdog,256);
title('The histogram of histogram equalization image')
%% 

%%%1.7%%%
ourimg=imread('flower.jpg');
ourimggray=double(rgb2gray(ourimg));
normourimggray=dip_GN_imread('flower.jpg');

city=imread('city.jpg');
citygray=double(rgb2gray(city));
normcitygray=dip_GN_imread('city.jpg');

face=imread('face.jpg');
facegray=double(face);
normfacegray=(facegray-min(facegray(:)))./(max(facegray(:))-min(facegray(:)));

figure;
subplot(3,2,1);
imshow(normourimggray);
title('Our Image');
colorbar
subplot(3,2,2);
dip_histogram(256*normourimggray,256);
title('Histogram of Our Image')

subplot(3,2,3);
imshow(normcitygray);
title('City Image');
colorbar
subplot(3,2,4);
dip_histogram(256*normcitygray,256);
title('Histogram of City Image')

subplot(3,2,5);
imshow(normfacegray);
title('Face Image');
colorbar
subplot(3,2,6);
dip_histogram(256*normfacegray,256);
title('Histogram of face Image')

histcity=imhistmatch(normourimggray, normcitygray);
histface=imhistmatch(normourimggray, normfacegray);

figure;
subplot(2,2,1);
imshow(histcity);
title('Our Image and City Match');
colorbar
subplot(2,2,2);
dip_histogram(256*histcity,256);
title('Histogram of Our Image and City Match')

subplot(2,2,3);
imshow(histface);
title('Our Image and Face Match');
colorbar
subplot(2,2,4);
dip_histogram(256*histface,256);
title('Histogram of Our Image and Face Match')

%% 

%%%%%-----------------2.1--------------------

Image = imread("dog.jpg");
imshow(Image)
Im_grey_norm = dip_GN_imread("dog.jpg")
imshow(Im_grey_norm)
colorbar
title("dog normailized in grayscale")
%% 

%%%%%-----------------2.2--------------------

figure;
meaned_img_3 = mean_filter(Im_grey_norm,3);
median_img_3 = median_filter(Im_grey_norm,3);
a_3=subplot(2,1,1);
imshow(meaned_img_3)
title("mean filter k=3")
b_3=subplot(2,1,2);
imshow(median_img_3)
title("median filter k=3")


figure;
meaned_img_5 = mean_filter(Im_grey_norm,5);
median_img_5 = median_filter(Im_grey_norm,5);
a_5=subplot(2,1,1);
imshow(meaned_img_5)
title("mean filter k=5")
b_5=subplot(2,1,2);
imshow(median_img_5)
title("median filter k=5")



figure;
meaned_img_9 = mean_filter(Im_grey_norm,9);
median_img_9 = median_filter(Im_grey_norm,9);
a_9=subplot(2,1,1);
imshow(meaned_img_9)
title("mean filter k=9")
b_9=subplot(2,1,2);
imshow(median_img_9)
title("median filter k=9")




%% 

%%%%%-----------------2.3--------------------

figure;

Gaussian_img_3_02=dip_gaussian_filter(Im_grey_norm,3,0.2);
c_1=subplot(2,2,1);
imshow(Gaussian_img_3_02);
title("Gaussian filter (3,0.2)")


c_2=subplot(2,2,2);
Gaussian_img_3_17=dip_gaussian_filter(Im_grey_norm,3,1.7);
imshow(Gaussian_img_3_17);
title("Gaussian filter (3,1.7)")

c_3=subplot(2,2,3);
Gaussian_img_9_02=dip_gaussian_filter(Im_grey_norm,9,0.2);
imshow(Gaussian_img_9_02);
title("Gaussian filter (9,0.2)")

c_4=subplot(2,2,4);
Gaussian_img_9_17=dip_gaussian_filter(Im_grey_norm,9,1.7);
imshow(Gaussian_img_9_17);
title("Gaussian filter (9,1.7)")

figure;

sub_Gaussian_img_3_02=Im_grey_norm-Gaussian_img_3_02;
c_5=subplot(2,2,1);
imshow(sub_Gaussian_img_3_02);
title("subtract Gaussian filter (3,0.2)")


c_6=subplot(2,2,2);
sub_Gaussian_img_3_17=Im_grey_norm-Gaussian_img_3_17;
imshow(sub_Gaussian_img_3_17);
title("subtract Gaussian filter (3,1.7)")

c_7=subplot(2,2,3);
sub_Gaussian_img_9_02=Im_grey_norm-Gaussian_img_9_02;
imshow(sub_Gaussian_img_9_02);
title("subtract Gaussian filter (9,0.2)")

c_8=subplot(2,2,4);
sub_Gaussian_img_9_17=Im_grey_norm-Gaussian_img_9_17;
imshow(sub_Gaussian_img_9_17);
title("subtract Gaussian filter (9,1.7)")
%% 

%%%%%-----------------2.4--------------------


anisodiff2D(Im_grey_norm,15, 1/7, 30, 2);
title("anisodiff2(15,1/7,30,2)")

%% 

%%%%%-----------------2.5.1---------------------

Im_salt_pepper=imnoise(Im_grey_norm,"salt & pepper");
Im_gaussian=imnoise(Im_grey_norm,"gaussian");
Im_speckle=imnoise(Im_grey_norm,"speckle");
%% 

%%%%%-----------------2.5.2-2.5.3---------------------

%%%%%-----------------dog salt and pepper--------------------
figure;
d_2=subplot(3,3,2);
imshow(Im_salt_pepper);
title("salt & pepper noise")

d_4=subplot(3,3,4);
imshow(mean_filter(Im_salt_pepper,3));
title("Mean filter 3X3")

d_7=subplot(3,3,7);
imshow(mean_filter(Im_salt_pepper,9));
title("Mean filter 9X9")


d_5=subplot(3,3,5);
imshow(median_filter(Im_salt_pepper,3));
title("Median filter 3X3")


d_8=subplot(3,3,8);
imshow(median_filter(Im_salt_pepper,9));
title("Median filter 9X9")


d_6=subplot(3,3,6);
imshow(dip_gaussian_filter(Im_salt_pepper,3,1));
title("gaussian filter 3X3, sigma = 1")


d_9=subplot(3,3,9);
imshow(dip_gaussian_filter(Im_salt_pepper,9,1));
title("gaussian filter 9X9, sigma = 1")


anisodiff2D(Im_salt_pepper,15, 1/7, 30, 2);
title("salt & pepper  - anisotropic diffusion")

%%%%%-----------------dog gaussian noise--------------------

figure;

e_2=subplot(3,3,2);
imshow(Im_gaussian);
title("gaussian noise")

e_4=subplot(3,3,4);
imshow(mean_filter(Im_gaussian,3));
title("Mean filter 3X3")

e_7=subplot(3,3,7);
imshow(mean_filter(Im_gaussian,9));
title("Mean filter 9X9")


e_5=subplot(3,3,5);
imshow(median_filter(Im_gaussian,3));
title("Median filter 3X3")


e_8=subplot(3,3,8);
imshow(median_filter(Im_gaussian,9));
title("Median filter 9X9")


e_6=subplot(3,3,6);
imshow(dip_gaussian_filter(Im_gaussian,3,1));
title("gaussian filter 3X3, sigma = 1")


e_9=subplot(3,3,9);
imshow(dip_gaussian_filter(Im_gaussian,9,1));
title("gaussian filter 9X9, sigma = 1")


anisodiff2D(Im_gaussian,15, 1/7, 30, 2);
title("gaussian  - anisotropic diffusion");


%%%%%-----------------dog speckle noise--------------------

figure;
f_2=subplot(3,3,2);
imshow(Im_speckle);
title("speckle noise")

f_4=subplot(3,3,4);
imshow(mean_filter(Im_speckle,3));
title("Mean filter 3X3")

f_7=subplot(3,3,7);
imshow(mean_filter(Im_speckle,9));
title("Mean filter 9X9")


f_5=subplot(3,3,5);
imshow(median_filter(Im_speckle,3));
title("Median filter 3X3")


f_8=subplot(3,3,8);
imshow(median_filter(Im_speckle,9));
title("Median filter 9X9")


f_6=subplot(3,3,6);
imshow(dip_gaussian_filter(Im_speckle,3,1));
title("gaussian filter 3X3, sigma = 1")


f_9=subplot(3,3,9);
imshow(dip_gaussian_filter(Im_speckle,9,1));
title("gaussian filter 9X9, sigma = 1")

figure;
anisodiff2D(Im_speckle,15, 1/7, 30, 2);
title("speckle  - anisotropic diffusion")

%%%%%-----------------2.5.4 square noises-----------------------------------


img_square= imread("square.jpg");
img_square=double(img_square);
norm_square=(img_square-min(img_square(:)))./(max(img_square(:))-min(img_square(:)))

sq_salt_pepper=imnoise(norm_square,"salt & pepper");
sq_gaussian=imnoise(norm_square,"gaussian");
sq_speckle=imnoise(norm_square,"speckle");

%%%%%-----------------square salt and pepper noise--------------------

figure;
d_2=subplot(3,3,2);
imshow(sq_salt_pepper);
title("salt & pepper noise")

d_4=subplot(3,3,4);
imshow(mean_filter(sq_salt_pepper,3));
title("Mean filter 3X3")

d_7=subplot(3,3,7);
imshow(mean_filter(sq_salt_pepper,9));
title("Mean filter 9X9")


d_5=subplot(3,3,5);
imshow(median_filter(sq_salt_pepper,3));
title("Median filter 3X3")


d_8=subplot(3,3,8);
imshow(median_filter(sq_salt_pepper,9));
title("Median filter 9X9")


d_6=subplot(3,3,6);
imshow(dip_gaussian_filter(sq_salt_pepper,3,1));
title("gaussian filter 3X3, sigma = 1")


d_9=subplot(3,3,9);
imshow(dip_gaussian_filter(sq_salt_pepper,9,1));
title("gaussian filter 9X9, sigma = 1")


anisodiff2D(sq_salt_pepper,15, 1/7, 30, 2);
title("salt & pepper  - anisotropic diffusion")

%%%%%-----------------square gaussian noise--------------------

figure;

l_2=subplot(3,3,2);
imshow(sq_gaussian);
title("gaussian noise")

l_4=subplot(3,3,4);
imshow(mean_filter(sq_gaussian,3));
title("Mean filter 3X3")

l_7=subplot(3,3,7);
imshow(mean_filter(sq_gaussian,9));
title("Mean filter 9X9")


l_5=subplot(3,3,5);
imshow(median_filter(sq_gaussian,3));
title("Median filter 3X3")


l_8=subplot(3,3,8);
imshow(median_filter(sq_gaussian,9));
title("Median filter 9X9")


l_6=subplot(3,3,6);
imshow(dip_gaussian_filter(sq_gaussian,3,1));
title("gaussian filter 3X3, sigma = 1")


l_9=subplot(3,3,9);
imshow(dip_gaussian_filter(sq_gaussian,9,1));
title("gaussian filter 9X9, sigma = 1")


anisodiff2D(sq_gaussian,15, 1/7, 30, 2);
title("gaussian  - anisotropic diffusion");


%%%%%-----------------square speckle noise--------------------

figure;
f_2=subplot(3,3,2);
imshow(sq_speckle);
title("speckle noise")

f_4=subplot(3,3,4);
imshow(mean_filter(sq_speckle,3));
title("Mean filter 3X3")

f_7=subplot(3,3,7);
imshow(mean_filter(sq_speckle,9));
title("Mean filter 9X9")


f_5=subplot(3,3,5);
imshow(median_filter(sq_speckle,3));
title("Median filter 3X3")


f_8=subplot(3,3,8);
imshow(median_filter(sq_speckle,9));
title("Median filter 9X9")


f_6=subplot(3,3,6);
imshow(dip_gaussian_filter(sq_speckle,3,1));
title("gaussian filter 3X3, sigma = 1")


f_9=subplot(3,3,9);
imshow(dip_gaussian_filter(sq_speckle,9,1));
title("gaussian filter 9X9, sigma = 1")


anisodiff2D(sq_speckle,15, 1/7, 30, 2);
title("speckle  - anisotropic diffusion")
%% 

%%%%%----------------- 2.5.5 graphs--------------------
figure;
plots(sq_salt_pepper)
title("salt pepper noise")
figure;
plots(sq_gaussian)
title("gaussian noise")
figure;
plots(sq_speckle)
title("speckle noise")

figure;
h_1=subplot(1,3,1);
plots(mean_filter(sq_salt_pepper,3))
title("mean")
h_2=subplot(1,3,2);
plots(median_filter(sq_salt_pepper,3))
title("median")
h_3=subplot(1,3,3);
plots(dip_gaussian_filter(sq_salt_pepper,9,1))
title("gaussian")
figure;
[a,b]=anisodiff2D(sq_salt_pepper, 15, 1/7, 30, 2)
figure;
plots(a)
title("anisotropic diffusion salt pepper")


figure;
h_1=subplot(1,3,1);
plots(mean_filter(sq_gaussian,3))
title("mean")
h_2=subplot(1,3,2);
plots(median_filter(sq_gaussian,3))
title("median")
h_3=subplot(1,3,3);
plots(dip_gaussian_filter(sq_gaussian,9,1))
title("gaussian")
figure;
[a,b]=anisodiff2D(sq_gaussian, 15, 1/7, 30, 2)
figure;
plots(a)
title("anisotropic diffusion gaussian")


figure;
h_1=subplot(1,3,1);
plots(mean_filter(sq_speckle,3))
title("mean")
h_2=subplot(1,3,2);
plots(median_filter(sq_speckle,3))
title("median")
h_3=subplot(1,3,3);
plots(dip_gaussian_filter(sq_speckle,9,1))
title("gaussian")
figure;
[a,b]=anisodiff2D(sq_speckle, 15, 1/7, 30, 2)
figure;
plots(a)
title("anisotropic diffusion speckle")


