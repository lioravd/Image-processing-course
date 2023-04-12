%Lior Avadyayev_206087611 Ayalla Reuven_314077033
function [brightimg]=adjust_brightness(img,action,parameter)    %1.3 Brightness
if action=='mul'

    brightimg=img.*parameter;
end
if action=='add'
    brightimg=img+parameter;
end
brightimg(brightimg>1)=1;
brightimg(brightimg<0)=0;
end