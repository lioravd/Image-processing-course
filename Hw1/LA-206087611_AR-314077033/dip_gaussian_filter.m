%Lior Avadyayev_206087611 Ayalla Reuven_314077033
function [img_gaussian] = dip_gaussian_filter(img,k,sigma) %gaussian filter
lim=floor(k/2);
x= -lim:1:lim;
y= -lim:1:lim;
[Filter_x,Filter_y]=meshgrid(x,y);
Gaussian_filter=(1/(2*pi*sigma^2)).*exp(-(Filter_x.^2+Filter_y.^2)./(2*sigma^2));
Gaussian_filter= Gaussian_filter./sum(Gaussian_filter,"all");
fil_conv = conv2(Gaussian_filter,img);
[hei,wid]= size(fil_conv);
img_gaussian=fil_conv(lim+1:hei-lim,lim+1:wid-lim);
end