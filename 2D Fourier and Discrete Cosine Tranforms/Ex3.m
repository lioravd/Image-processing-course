clc;
clear all;
disp('Ayalla Reuven 314077033')
disp('Lior Avadyayev 206087611')
%%
%%%1.1
%1.1.3
beatles=dip_GN_imread("beatles.png");   %convert beatles image to grayscale normalized image.
figure;
imshow(beatles); 
title('Q1.1.3: Beatles Image');

%1.1.4
FFT_beatles= dip_fft2(beatles);     %Compute the 2D-FFT of the image
shift_beatles= dip_fftshift(FFT_beatles);   % shift the output image
figure;
subplot(1,2,1)
imagesc(log(abs(shift_beatles)));       %log|amplitude|
title('Q1.1.4: log of the amplitude of FFT shift grayscale normalized Beatles');
colorbar;
subplot(1,2,2)
imagesc(angle(shift_beatles));      %phase
title('Q1.1.4: Phase of FFT shift grayscale normalized Beatles')
colorbar;

%1.1.5
rec_beatles=dip_ifft2(dip_fftshift(shift_beatles));     %Reconstruct the original image
error= sum(sum(abs(beatles-rec_beatles)))      %Chece if it is identical to the original image
figure;
subplot(1,2,1)
imshow(beatles); 
title('Q1.1.5: Original Beatles Image');
subplot(1,2,2)
imshow(real(rec_beatles));
title('Q1.1.5: Reconstruct the Beatles Image');
%%
%%%1.2
%1.2.1
load('freewilly.mat'); 
figure;
imshow(freewilly);
title('Q1.2.1(a): Willie behind bars');

figure;
plot(freewilly(1,:));   %plot the first row of the image 
title('Q1.2.1(b): Willy first row');
grid on;

fx= size(findpeaks(freewilly(1,:)),2);      %caculate fx

[M,N]= size(freewilly);
x = meshgrid(0:N-1,0:M-1);
bars = 0.5*sin((2*pi*fx/N)*x);
figure; 
imshow(bars);
title('Q1.2.1(c): Prison Bar')

shifFFT_bars= dip_fftshift(dip_fft2(bars));   
figure;
imagesc(abs(shifFFT_bars));       %log|amplitude|
title('Q1.2.1(d): Amplitude of FFT shift of Prison Bar');

Free_Willy(freewilly);
%%
%1.2.2
matrix1=zeros(128,128);     %(a) Create the first square
matrix1(44:83,44:83) = 1;
shift_matrixFFT=dip_fftshift(dip_fft2(matrix1));

figure;     %(a) Display the first square
imshow(matrix1);
title('Q1.2.2(a): Matrix Image 1- Square')
figure;
subplot(1,2,1)      %(a) Display the FFT of first square
imagesc(abs(shift_matrixFFT));
title('Q1.2.2(a): |FFT(Matrix Image 1- Square)|')
subplot(1,2,2)
imagesc(angle(shift_matrixFFT));
title('Q1.2.2(a): Phase of FFT(Matrix Image - Square)')

matrix2=zeros(128,128);     %(b) Create the second square
matrix2(64:103,64:103)=1;
shift_matrixFFT2=dip_fftshift(dip_fft2(matrix2));

figure;     %(b) Display the second square
imshow(matrix2);
title('Q1.2.2(b): Matrix Image 2 - Square')
figure;
subplot(1,2,1)         %(b) Display the FFT of second square
imagesc(abs(shift_matrixFFT2));
title('Q1.2.2(b): |FFT(Matrix Image 2- Side Square)|')
subplot(1,2,2)
imagesc(angle(shift_matrixFFT2));
title('Q1.2.2(b): Phase of FFT(Matrix Image 2- Square)')

figure;     %(b) Compare between the FFT amplitude of the previuos and the new square
subplot(1,2,1)
imagesc(abs(shift_matrixFFT));
title('Q1.2.2(b): |FFT(Matrix Image 1- Square)|)')
subplot(1,2,2)
imagesc(abs(shift_matrixFFT2));
title('Q1.2.2(b): |FFT(Matrix Image 2- Square)|')

matrix3=zeros(128,128);     %(c) Create the rectangle
matrix3(24:103,54:73)=1;
shift_matrixFFT3=dip_fftshift(dip_fft2(matrix3));

figure;             %(c) Display the rectangle
imshow(matrix3);
title('Q1.2.2(c): Matrix Image 3 - Rectangle')
figure;             %Display the FFT of rectangle
subplot(1,2,1)
imagesc(abs(shift_matrixFFT3));
title('Q1.2.2(c): |FFT(Matrix Image 3- Rectangle)|')
subplot(1,2,2)
imagesc(angle(shift_matrixFFT3));
title('Q1.2.2(c): Phase of FFT(Matrix Image 3- Rectangle)')

figure;             %(c) Compare between the FFT amplitude of the previuos square and the new rectangle
subplot(1,2,1)
imagesc(abs(shift_matrixFFT));
title('Q1.2.2(b): |FFT(Matrix Image 1- Square)|')
subplot(1,2,2)
imagesc(abs(shift_matrixFFT3));
title('Q1.2.2(b): |FFT(Matrix Image 3 - Rectangle)|')

vec1=zeros(128,1);           %(d) Create the rectangle using two 1D vectors
vec2=zeros(128,1);
vec1(24:103)=1;
vec2(54:73)=1;
figure;
imshow(vec1*vec2'); 
title('Q1.2.2(d): Rectangle using two 1D vectors');

fft_2vec=sep_fft2(vec1',vec2');           %(e) compute 2D-FFT of an image using 1D-FFTs when the image is separable into two 1D vectors
shift_fft2vec=dip_fftshift(fft_2vec);
figure;                 %(e) Compare between the 2D-FFT using 1D-FFTsx and the previuos 2D-FFT
subplot(1,2,1)
imagesc(abs(shift_matrixFFT3));
title('Q1.2.2(e): |FFT(Matrix Image 1- Square)| using fft2 ')
subplot(1,2,2)
imagesc(abs(shift_fft2vec));
title('Q1.2.2(e): FFT(|Matrix Image 2- Square|) using 1D-FFT')
error=sum(sum(abs(abs(shift_fft2vec)-abs(shift_matrixFFT3))));
%% 

%2
%2.2
dct_beatles= dct2(beatles);  %Compute DCT to beatles
figure;
imagesc(log(abs(dct_beatles)))
colormap(jet(64))
colorbar
title("2.2 beatles DCT")

%2.3
[M,N]=size(dct_beatles);
dct_beatles_50_zeros = binornd(1,0.5,[M,N]).*dct_beatles;   %create M*N binary Binomial matrix and multipy
idct_beatles_50_zeros = idct2(dct_beatles_50_zeros);
figure;
imshow(idct_beatles_50_zeros)
title("2.3 IDCT with 50% zeros")

%2.4
dct_beatles_50_smallest = dct_beatles;
median_dct = median(abs(dct_beatles),"all");
dct_beatles_50_smallest(abs(dct_beatles)<median_dct)=0;
idct_beatles_50_smallest = idct2(dct_beatles_50_smallest);
figure;
imshow(idct_beatles_50_smallest)
title("2.4 IDCT with 50% smallest")


%2.5
dct_beatles_range_005 = dct_beatles;
dct_beatles_range_005(abs(dct_beatles)<0.05)=0;
idct_beatles_range_005 = idct2(dct_beatles_range_005);

dct_beatles_range_01 = dct_beatles;
dct_beatles_range_01(abs(dct_beatles)<0.1)=0;
idct_beatles_range_01 = idct2(dct_beatles_range_01);

dct_beatles_range_02 = dct_beatles;
dct_beatles_range_02(abs(dct_beatles)<0.2)=0;
idct_beatles_range_02 = idct2(dct_beatles_range_02);

dct_beatles_range_03 = dct_beatles;
dct_beatles_range_03(abs(dct_beatles)<0.3)=0;
idct_beatles_range_03 = idct2(dct_beatles_range_03);

dct_beatles_range_04 = dct_beatles;
dct_beatles_range_04(abs(dct_beatles)<0.4)=0;
idct_beatles_range_04 = idct2(dct_beatles_range_04);

dct_beatles_range_05 = dct_beatles;
dct_beatles_range_05(abs(dct_beatles)<0.5)=0;
idct_beatles_range_05 = idct2(dct_beatles_range_05);

figure;
subplot(3,2,1)
imshow(idct_beatles_range_005)
title("dct a=0.05")
subplot(3,2,2)
imshow(idct_beatles_range_01)
title("dct a=0.1")
subplot(3,2,3)
imshow(idct_beatles_range_02)
title("dct a=0.2")
subplot(3,2,4)
imshow(idct_beatles_range_03)
title("dct a=0.3")
subplot(3,2,5)
imshow(idct_beatles_range_04)
title("dct a=0.4")
subplot(3,2,6)
imshow(idct_beatles_range_05)
title("dct a=0.5")

count_per=0;
for i=1:M
    for j=1:N
        if dct_beatles_range_04(i,j)==0
            count_per=count_per+1;
        end
    end
end

count_per=count_per/(M*N);
%% 

%3.1
beetle=dip_GN_imread("beetle.jpg");
figure;
imshow(beetle); 
title('Beetle Image');
colorbar


%3.2 

[wave_beetle_c,wave_beetle_s] = wavedec2(beetle,3,"haar");   %Wavelet extraction


%3.3
[H1,V1,D1] = detcoef2('all',wave_beetle_c,wave_beetle_s,1);   
A1 = appcoef2(wave_beetle_c,wave_beetle_s,'haar',1);

[H2,V2,D2] = detcoef2('all',wave_beetle_c,wave_beetle_s,2);
A2 = appcoef2(wave_beetle_c,wave_beetle_s,'haar',2);

[H3,V3,D3] = detcoef2('all',wave_beetle_c,wave_beetle_s,3);
A3 = appcoef2(wave_beetle_c,wave_beetle_s,'haar',3);

%3.4

figure;
subplot(2,2,1)
imagesc(A1)
colormap pink(255)
title('Approximation Coef. of Level 1')

subplot(2,2,2)
imagesc(H1)
title('Horizontal Detail Coef. of Level 1')

subplot(2,2,3)
imagesc(V1)
title('Vertical Detail Coef. of Level 1')

subplot(2,2,4)
imagesc(D1)
title('Diagonal Detail Coef. of Level 1')

figure;
subplot(2,2,1)
imagesc(A2)
colormap pink(255)
title('Approximation Coef. of Level 2')

subplot(2,2,2)
imagesc(H2)
title('Horizontal Detail Coef. of Level 2')

subplot(2,2,3)
imagesc(V2)
title('Vertical Detail Coef. of Level 2')

subplot(2,2,4)
imagesc(D2)
title('Diagonal Detail Coef. of Level 2')

figure;
subplot(2,2,1)
imagesc(A3)
colormap pink(255)
title('Approximation Coef. of Level 3')

subplot(2,2,2)
imagesc(H3)
title('Horizontal Detail Coef. of Level 3')

subplot(2,2,3)
imagesc(V3)
title('Vertical Detail Coef. of Level 3')

subplot(2,2,4)
imagesc(D3)
title('Diagonal Detail Coef. of Level 3')





%%
function [F]=dip_fft2(I)      %1.1
[M,N]=size(I);
Mvec=0:1:M-1;
Nvec=0:1:N-1;
Mexp=exp(-2*pi*(Mvec')*Mvec*1i/M);
Nexp=exp(-2*pi*(Nvec')*Nvec*1i/N);
F=Mexp*I*Nexp;
end

function [I]=dip_ifft2(F)     %1.1
[M,N]=size(F);
Mvec=0:1:M-1;
Nvec=0:1:N-1;
Mexp=exp(2*pi*(Mvec')*Mvec*1i/M);
Nexp=exp(2*pi*(Nvec')*Nvec*1i/N);
I=(1/(M*N))*Mexp*F*Nexp;
end

function [Fshift]=dip_fftshift(F)      %1.2
[N,M]=size(F);
Fshift=[F(:,floor(M/2)+1:end),F(:,1:floor(M/2))];                 
Fshift=[Fshift(floor(N/2)+1:end,:);Fshift(1:floor(N/2),:)]; 
end

function [free_willy]=Free_Willy(willy) 
fx=size(findpeaks(willy(1,:)),2);
[M,N]= size(willy);
x=meshgrid(0:N-1,0:M-1);
bars=0.5*sin((2*pi*fx/N)*x);
free_willy=willy-bars;

figure;
imshow(free_willy);
title('Free Willy');
end

function [F]=sep_fft2(v1,v2)
N1=length(v1);
N2=length(v2);
N1vec=0:1:N1-1;
N2vec=0:1:N2-1;
N1exp=exp(-2*pi*(N1vec')*N1vec*1i/N1);
N2exp=exp(-2*pi*(N2vec')*N2vec*1i/N2);
F=(N1exp*v1')*(N2exp*v2')';
end