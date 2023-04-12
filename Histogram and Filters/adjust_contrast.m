%Lior Avadyayev_206087611 Ayalla Reuven_314077033
function [contimg]=adjust_contrast(img,range_low,range_high)    %1.4  Contrast
contimg=((range_high-range_low).*img)+range_low;
end
