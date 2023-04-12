%Lior Avadyayev_206087611 Ayalla Reuven_314077033
function [normgray]=dip_GN_imread(file_name)    %1.1 normalized gray scale image
pic=imread(file_name);
pic=double(pic);
normgray=(pic-min(pic(:)))./(max(pic(:))-min(pic(:)));
end