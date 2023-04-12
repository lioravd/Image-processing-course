%Lior Avadyayev_206087611 Ayalla Reuven_314077033
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