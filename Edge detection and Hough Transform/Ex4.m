clc;
clear all;
disp('Ayalla Reuven 314077033')
disp('Lior Avadyayev 206087611')
%%
%1
%1.1
camera_man=dip_GN_imread("cameraman.tif");  %read camera and normalize
figure;
imshow(camera_man)
%%
%1.2
G_0 = dip_prewitt_edge(camera_man,0);    %prewitt edges
G_05 = dip_prewitt_edge(camera_man,0.2);   
figure;
subplot(1,2,1)
imshow(G_0)
title("Treshold = 0")
subplot(1,2,2)
imshow(G_05)
title("Treshold = 0.2")
%%
%1.3
figure;

a=max(a(:))
can_camera_def = edge(camera_man,"canny");   %canny edges

can_camera = edge(camera_man,'canny',[0.1 0.2]);
%%sum5= sum((abs(can_camera-can_camera_def)),"all")
subplot(1,2,1)
imshow(can_camera)
title("canny with [0.1 0.2] threshold")
subplot(1,2,2)
imshow(can_camera_def)
title("canny with default parametres")
%%
%2
%2.1

%a
floor  = dip_GN_gray_imread("floor.jpg");
figure;
imshow(floor)
title("Q2.1(a): Floor Image")

%b
BW_floor= edge(floor);
figure;
imshow(BW_floor)
title("Q2.1(b): default edge for floor")


%c
R0=1; teta0=1;
floor_hough_matrix1=dip_hough_lines(BW_floor,R0,teta0);
R0=5; teta0=4;
floor_hough_matrix2=dip_hough_lines(BW_floor,R0,teta0);

%d
figure;
imshow(floor_hough_matrix1,[]);
title("Q2.1(d): Hough matrix R0=1 teta0=1")
xlabel('\fontsize{20}\theta');
ylabel('\fontsize{20}\rho');
axis on
axis normal

figure;
imshow(floor_hough_matrix2,[]);
title("Q2.1(d):Hough matrix: R0=5 teta0=4")
xlabel('\fontsize{20}\theta');
ylabel('\fontsize{20}\rho');
axis on
axis normal

%e
%R0,T0=(1,1)
R0=1; teta0=1;
peaks = houghpeaks(floor_hough_matrix1,4,'Threshold',1);
[M,N] = size(BW_floor);
R = (-sqrt(M^2+N^2)):R0:(sqrt(M^2+N^2));
teta = (-90:teta0:90)*(pi/180);

m= -1 ./ tan(teta(peaks(:,2)));
x= R(peaks(:,1)).*cos(teta(peaks(:,2)));
y= R(peaks(:,1)).*sin(teta(peaks(:,2)));
b= y-m.*x;
Xaxis= 1:N;

figure;
imshow(floor);
title("Q2.1(e): Floor Hough Matrix for R0=1, teta0 =1");
for i = 1:length(m)
    hold on;
    plot(Xaxis,m(i)*Xaxis+b(i),'LineWidth',1,'color','m');
    axis on
end

%R0,T0=(5,4)
R0=5; teta0=4;
peaks = houghpeaks(floor_hough_matrix2,4,'Threshold',1);
[M,N] = size(BW_floor);
R = (-sqrt(M^2+N^2)):R0:(sqrt(M^2+N^2));
teta = (-90:teta0:90)*(pi/180);

m= -1 ./ tan(teta(peaks(:,2)));
x= R(peaks(:,1)).*cos(teta(peaks(:,2)));
y= R(peaks(:,1)).*sin(teta(peaks(:,2)));
b= y-m.*x;
Xaxis= 1:N;

figure;
imshow(floor);
title("Q2.1(e): Floor Hough Matrix for R0=5, teta0 =4");
for i = 1:length(m)
    hold on;
    plot(Xaxis,m(i)*Xaxis+b(i),'LineWidth',1,'color','m');
    axis on
end
%%
%2.2
%a
coffee = dip_GN_gray_imread("coffee.jpg");
figure;
imshow(coffee)
title("Q2.2(a): Coffee Image")

%b
BW_coffee= edge(coffee);
figure;
imshow(BW_coffee)
title("Q2.2(b): default edge for coffee")

%c
tic;
R0=1; teta0=1;
coffee_hough_matrix1=dip_hough_circles(BW_coffee,R0,teta0);
toc;
R0=4; teta0=10;
coffee_hough_matrix2=dip_hough_circles(BW_coffee,R0,teta0);

%d
tic;
R0=1; teta0=5;
coffee_hough_matrix3=dip_hough_circles(BW_coffee,R0,teta0);
toc;   

%e
figure;
imshow(coffee_hough_matrix1(:,:,1),[]);
axis normal
title('Q2.2(e): Coffee Hough Matrix for R0 = 1, teta0 = 1');
axis on
xlabel('a');
ylabel('b');

figure;
imshow(coffee_hough_matrix2(:,:,1),[]);
axis normal
title('Q2.2(e): Coffee Hough Matrix for R0 = 4 teta0 = 10');
axis on
xlabel('a');
ylabel('b');

figure;
imshow(coffee_hough_matrix3(:,:,1),[]);
axis normal
title('Q2.2(e): Coffee Hough Matrix for R0 =1 teta0 = 5');
axis on
xlabel('a')
ylabel('b');

%f
%R0,T0=(1,1)
R0=1;
peaks = dip_houghpeaks3d(coffee_hough_matrix1);
R = 80:R0:100;

figure;
imshow(coffee);
title("Q2.2(f):Coffee Hough Matrix for R0=1, teta0 =1");
for i = 1:5
    hold on
    circle(peaks(i,1),peaks(i,2),R(peaks(i,3)));
    axis on
end

%R0,T0=(4,10)
R0=4;
peaks = dip_houghpeaks3d(coffee_hough_matrix2);
R = 80:R0:100;

figure;
imshow(coffee);
title("Q2.2(f): Coffee Hough Matrix for R0=4, teta0 =10");
for i = 1:5
    hold on
    circle(peaks(i,1),peaks(i,2),R(peaks(i,3)));
    axis on
end

%R0,T0=(1,5)
R0=1;
peaks = dip_houghpeaks3d(coffee_hough_matrix3);
R = 80:R0:100;

figure;
imshow(coffee);
title("Q2.2(f): Coffee Hough Matrix for R0=1, teta0 =5");
for i = 1:5
    hold on
    circle(peaks(i,1),peaks(i,2),R(peaks(i,3)));
    axis on
end
%%
function[edge] = dip_prewitt_edge(img,tresh)
GP_x= (1/6).*[[-1 0 1]; [-1 0 1]; [-1 0 1]];
GP_y= (1/6).*[[1 1 1] ;[0 0 0]; [-1 -1 -1]];   
der_x = conv2(img,GP_x,"same");
der_y = conv2(img,GP_y,"same");
edge = sqrt(der_x.^2+der_y.^2);
edge(edge<tresh)=0;
end

function [normgray]=dip_GN_gray_imread(file_name)    %1.1 normalized gray scale image
pic=imread(file_name);
picgray = double(rgb2gray(pic));
normgray=(picgray-min(picgray(:)))./(max(picgray(:))-min(picgray(:)));
end

function [hough_matrix]=dip_hough_lines(BW,R0,teta0)
[M,N]= size(BW);
R=(-sqrt(M^2+N^2)):R0:(sqrt(M^2+N^2));
teta= -90:teta0:90;
abs_R=length(R);
abs_teta=length(teta);
hough_matrix=zeros(abs_R,abs_teta);
[rows,columns] = find(BW);
for i=1:length(rows)
    x=columns(i); y=rows(i);
    for t=1:abs_teta
        r=x*cos(teta(t)*pi/180)+y*sin(teta(t)*pi/180);
        [~,I] = min(abs(R-r),[],2);
        hough_matrix(I,t)=hough_matrix(I,t)+1;

    end
end
end
                                                                                                                                                                                                                       
function [hough_matrix]=dip_hough_circles(BW,R0,teta0)  %Q2.2(c)  Hough Matrix for finding circles.
[M,N]= size(BW);
R=80:R0:100;
teta= 0:teta0:360;
abs_R=length(R);
hough_matrix=zeros(M,N,abs_R);
[rows,columns] = find(BW);
for i=1:length(rows)
    x=columns(i); y=rows(i);
    for r=1:abs_R
        for t=1:length(teta)
            a=x-R(r)*cos(teta(t)*(2*pi/360));
            b=y-R(r)*sin(teta(t)*(2*pi/360));
            a=round(a); b=round(b);
            if a<=M && a>0 && b>0 && b<=N
                hough_matrix(a,b,r)=hough_matrix(a,b,r)+1;
            end
        end
    end
end
end

function [peaks]=dip_houghpeaks3d(HoughMatrix)     %Q2.2(f) find the 5 most significant circles 
H=HoughMatrix;
peaks = zeros(5,3);
for i = 1:5
    [~,idx] = max(H(:));
    [idx1,idx2,idx3] = ind2sub(size(H),idx);
    peaks(i,:) = [idx1,idx2,idx3];
    H(idx1+(-80:1:80),idx2+(-80:1:80),:)=0;
end
end

function h = circle(x,y,r)  %Q2.2(f) plot circle
hold on
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
h = plot(xunit, yunit,'Color','m','LineWidth',1.5);
hold off
end