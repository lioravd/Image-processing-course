close all
clear all
clc


%% Q 2.1

A = dip_GN_imread('dog.jpg');
imshow(A);
title('Grayscale dog - Q2.1')
%% Q 2.2
A = dip_GN_imread('dog.jpg');

%2.2
figure('Name','Q2.2.1');
subplot(2,3,1);
imshow(mean_filter(A,3));
title('Dog image with mean filter for k=3');
subplot(2,3,2);
imshow(mean_filter(A,5));
title('Dog image with mean filter for k=5');
subplot(2,3,3);
imshow(mean_filter(A,9));
title('Dog image with mean filter for k=9');

%2.2.2
figure('Name','Q2.2.2');
subplot(2,3,1);
imshow(median_filter(A,3));
title('Dog image with median filter for k=3');
subplot(2,3,2);
imshow(median_filter(A,5));
title('Dog image with median filter for k=5');
subplot(2,3,3);
imshow(median_filter(A,9));
title('Dog image with median filter for k=9');

%% Q 2.3
% 2.3.1
%  Create the function that makes gaussian filter on the image.
% 2.3.2

A = dip_GN_imread('dog.jpg');

figure('Name','Q2.3.1');
subplot(2,2,1)
imshow(dip_gaussian_filter(A,3,0.2));
title('Dog image with gaussian filter for k=3 and sigma=0.2');
subplot(2,2,2)
imshow(dip_gaussian_filter(A,3,1.7));
title('Dog image with gaussian filter for k=3 and sigma=1.7');
subplot(2,2,3)
imshow(dip_gaussian_filter(A,9,0.2));
title('Dog image with gaussian filter for k=9 and sigma=0.2');
subplot(2,2,4)
imshow(dip_gaussian_filter(A,9,1.7));
title('Dog image with gaussian filter for k=9 and sigma=1.7');

% 2.3.3

figure('Name','Q2.3.1');
subplot(2,2,1)
imshow(A - dip_gaussian_filter(A,3,0.2));
title('Dog image with gaussian filter for k=3 and sigma=0.2');
subplot(2,2,2)
imshow(A - dip_gaussian_filter(A,3,1.7));
title('Dog image with gaussian filter for k=3 and sigma=1.7');
subplot(2,2,3)
imshow(A - dip_gaussian_filter(A,9,0.2));
title('Dog image with gaussian filter for k=9 and sigma=0.2');
subplot(2,2,4)
imshow(A - dip_gaussian_filter(A,9,1.7));
title('Dog image with gaussian filter for k=9 and sigma=1.7');

%% Q2.4

A = dip_GN_imread('dog.jpg');
anisodiff2D(A, 15, 1/7, 30, 1, 8);
anisodiff2D(A, 15, 1/7, 30, 2, 9);



%% Q2.5
% Q 2.5.1-3

A = dip_GN_imread('dog.jpg');

salt_pepper = imnoise(A,"salt & pepper");
gaussian = imnoise(A,"gaussian");
speckle = imnoise(A,"speckle");

%display the noisy image
figure('Name','Q2.5.3');
subplot(2,3,1);
imshow(salt_pepper);
title('Dog image with salt&pepper noise');
subplot(2,3,2);
imshow(gaussian);
title('Dog image with gaussian noise');
subplot(2,3,3);
imshow(speckle);
title('Dog image with speckle noise');

%display the filters on the noist images, we built a function for
%functionallity of our code
 dip_all_filters(salt_pepper, 'salt & pepper');
 dip_all_filters(gaussian, 'gaussian');
 dip_all_filters(speckle, 'speckle');


% the anisotropic filter for salt_pepper noise image
anisodiff2D(salt_pepper, 10, 1/7, 30, 1,11);
anisodiff2D(salt_pepper, 10, 1/7, 30, 2,12);
% the anisotropic filter for gaussian noise image
anisodiff2D(gaussian, 10, 1/7, 30, 1,13);
anisodiff2D(gaussian, 10, 1/7, 30, 2,14);
% the anisotropic filter for speckle noise image
anisodiff2D(speckle, 10, 1/7, 30, 1,15);
anisodiff2D(speckle, 10, 1/7, 30, 2,16);
% the anisotropic filter ......

%% Q2.5.4 - Reapet 1-3 with the sqiare image

B = dip_GN_imread('square.jpg'); % normalized between 0 and 1...
figure ('Name', 'Q2.5.4');
imshow(B);
title('The normalized square image');

sq_salt_pepper = imnoise(B,"salt & pepper");
sq_gaussian = imnoise(B,"gaussian");
sq_speckle = imnoise(B,"speckle");

%display the noisy square images
figure('Name','Q2.5.4');
subplot(2,3,1);
imshow(sq_salt_pepper);
title('square image with salt&pepper noise');
subplot(2,3,2);
imshow(sq_gaussian);
title('square image with gaussian noise');
subplot(2,3,3);
imshow(sq_speckle);
title('square image with speckle noise');

dip_all_filters(sq_salt_pepper, 'salt & pepper');
dip_all_filters(sq_gaussian, 'gaussian');
dip_all_filters(sq_speckle, 'speckle');

% the anisotropic filter for salt_pepper noise image
anisodiff2D(sq_salt_pepper, 10, 1/7, 30, 1,17);
anisodiff2D(sq_salt_pepper, 10, 1/7, 30, 2,18);
% the anisotropic filter for gaussian noise image
anisodiff2D(sq_gaussian, 10, 1/7, 30, 1,19);
anisodiff2D(sq_gaussian, 10, 1/7, 30, 2,20);
% the anisotropic filter for speckle noise image
anisodiff2D(sq_speckle, 10, 1/7, 30, 1,21);
anisodiff2D(sq_speckle, 10, 1/7, 30, 2,22);

%% Q2.5.5 Analyze result
B = dip_GN_imread('square.jpg'); % normalized between 0 and 1...
sq_salt_pepper = imnoise(B,"salt & pepper");
sq_gaussian = imnoise(B,"gaussian");
sq_speckle = imnoise(B,"speckle");

%display the 300 line in the noisy square images
figure('Name','Q2.5.5');
subplot(2,2,1);
plot(B(300,:));
title('\fontsize{18}The 300 line in the original picture of the square');
subplot(2,2,2);
plot(sq_salt_pepper(300,:));
title('\fontsize{18}The 300 line in the salt&pepper picture of the square')
subplot(2,2,3);
plot(sq_gaussian(300,:));
title('\fontsize{18}The 300 line in the gaussian picture of the square');
subplot(2,2,4);
plot(sq_speckle(300,:));
title('\fontsize{18}The 300 line in the speckle picture of the square')

analyze_result(sq_salt_pepper,300,'salt&pepper noise');
analyze_result(sq_gaussian,300,'gaussian noise');
analyze_result(sq_speckle,300,'speckle noise');


%% functions Q2

% 2.1

function GN_image = dip_GN_imread(file_name)
    gray_image = double(im2gray(imread(file_name)));                                       %reading image and converting to grey double
    GN_image = (gray_image - min(gray_image(:)))/(max(gray_image(:))-min(gray_image(:)));   %normelizing
end



% 2.2.1 Mean filter

function [mean_filter] = mean_filter(img,k) %the image and the dimension

mean_filter = zeros(size(img)); % the final result
row = length(img(:,1));
col = length(img(1,:));
img = [img ; repmat(img(row,:),k-1,1)]; 
new_img = [img repmat(img(:,col),1,k-1)];
%duplicte the frame of the image
for i = 1:row
    for j = 1:col
        mean_filter(i,j) = sum(sum(new_img(i:i+k-1,j:j+k-1).*(ones(k,k)*(1/k^2)))); % the dot prodcut duplication
    end 
end

end
% 2.2.2 Median filter

function [median_filter] = median_filter(img,k) %the image and the dimension

median_filter = zeros(size(img)); % the final result
row = length(img(:,1));
col = length(img(1,:));
img = [img ; repmat(img(row,:),k-1,1)]; 
new_img = [img repmat(img(:,col),1,k-1)];
%duplicte the frame of the image

for i = 1:row
    for j = 1:col
        med_img = new_img(i:i+k-1,j:j+k-1); % each iteration the function calculate the median of the neighboor of the new_img
        median_filter(i,j) = median(med_img(:));
    end
end

end

%  2.3 Gaussian Filter
function [gaussian_filter] = dip_gaussian_filter(img,k, sigma) %the image and the dimension

gaussian_filter = zeros(size(img)); % the final result
row = length(img(:,1));
col = length(img(1,:));
img = [img ; repmat(img(row,:),k-1,1)]; 
new_img = [img repmat(img(:,col),1,k-1)];
cov = sigma.*eye(2);


%the gaussian filter with meshgrid() - 

x = -(k-1)/2 :1: (k-1)/2;
y = x
[X,Y] = meshgrid(x,y); %vector of X,Y

PDF_gauss = (1/(2*pi*det(cov)))*exp(-(X.^2+Y.^2)/(2*sigma^2)); % the pdf function of Gaussian vectoors...
normalized_gauss = PDF_gauss/sum(PDF_gauss(:));

for i = 1:row
    for j = 1:col
        gaussian_filter(i,j) = sum(sum(new_img(i:i+k-1,j:j+k-1).*normalized_gauss));

    end
end
end

% 2.5 functions that helps to display the images (with the filters)
function [] = dip_all_filters(img,img_name)
    for k = 3:6:9
        disp(k);
        figure('Name','All filters images for k=' + string(k) +' ' + string(img_name));
        subplot(2,2,1);
        imshow(mean_filter(img, k));
        title('Mean filter images for k=' + string(k));
        subplot(2,2,2);
        imshow(median_filter(img, k));
        title('Median filter images for k=' + string(k));
        subplot(2,2,3);
        imshow(dip_gaussian_filter(img,k,0.2));
        title('Gaussian with sigma = 0.2 filter images for k=' + string(k));
        subplot(2,2,4);
        imshow(dip_gaussian_filter(img,k,1.7));
        title('Gaussian with sigma = 1.7 filter images for k=' + string(k));
    end
end

function [] = analyze_result(img,row_num,img_name)  % write the mean,median and gaussian filter for any image... the plos has to be on the __'th row that we gave in the function
      

      k = [3 9];
      j = 40;
      for i = 1:2
         disp(k);
         mean = mean_filter(img, k(i));
         med = median_filter(img,k(i));
         gaussfilter_02 = dip_gaussian_filter(img,k(i),0.2); % the sigma is different
         gaussfilter_17 = dip_gaussian_filter(img,k(i),1.7);
         anisotrop = anisodiff2D(img,15,1/7,30,1,j);   %%%%% Huston we have a problem %%%%
         j = j+1;
       
         figure('Name','All filters images for k=' + string(k(i)) + ' and the image with ' + string(img_name));
         subplot(3,2,1);
         plot(mean(row_num,:));
         title('\fontsize{14} The ' + string(row_num) + ' line after Mean filter images for k=' + string(k(i)));
         ylim(0:1);

         subplot(3,2,2);
         plot(med(row_num,:));
         title('\fontsize{14} The ' + string(row_num) + ' line after Median filter images for k=' + string(k(i)));
         ylim(0:1);

         subplot(3,2,3);
         plot(gaussfilter_02(row_num,:));
         title('\fontsize{14} The ' + string(row_num) + ' line after Gaussian flter with sigma = 0.2 filter images for k=' + string(k(i)));
         ylim(0:1);

         subplot(3,2,4);
         plot(gaussfilter_17(row_num,:));
         title('\fontsize{14} The ' + string(row_num) + ' line after Gaussian flter with sigma = 1.7 filter images for k=' + string(k(i)));
         ylim(0:1);

         subplot(3,2,5);             %%%%% Huston we have a problem %%%%
         plot(anisotrop(row_num,:));
         title('\fontsize{14} The ' + string(row_num) + ' line after anisotropic diffusion filter for default parameters');



    end
end
