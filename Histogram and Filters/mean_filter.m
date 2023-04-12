%Lior Avadyayev_206087611 Ayalla Reuven_314077033
function [img_mean] = mean_filter(img,k)  %mean filter
Filter = (1/(k^2)).*ones(k,k);
fil_conv = conv2(Filter,img);
cut = floor(k/2);
[hei,wid]= size(fil_conv);
img_mean=fil_conv(cut+1:hei-cut,cut+1:wid-cut);
end