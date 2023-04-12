%Lior Avadyayev_206087611 Ayalla Reuven_314077033
function [histogram]=dip_histogram(img,nbins)   %1.2 histogram of the image
histogram=zeros(256,1);
m=0;
for i=1:nbins
    mat=zeros(size(img));
    mat(img>=(255*((i-1.5)/(nbins-1))-0) & img<(255*((i-0.5)/(nbins-1))-0))=1; 
    histogram(uint8(m+1))=sum(mat,"all");
    m=m+255/(nbins-1);
end
stem(0:255,histogram,'-','.')
end