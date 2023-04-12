function [score,  y, x ] = Generalized_hough_transform_standalone(Is,Itm) 
%Find template/shape Itm in greyscale image Is using generalize hough trasform
%show the image with the template marked on it
%Use generalized hough transform to find Template/shape binary image given in binary image Itm inimage Is (greyscale image)
%Return the x,y location  cordniates  which gave the best match 
%Also return the score of the match (number of point matching)

%INPUT
%Is is greyscale  picture were the template Itm should be found 
%Itm is binary edge image of the template with edges markedd ones


% OUTPUT
%Score of the best match
%x,y location of template Itm in image Is  (Location the edge (point [1,1]) of the template Itm in Is)

%%%%%%%%%%%%%%%%%%%%%%%%Reads images if this were not input%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (nargin<2) %if no image was given as input read images from file
Itm=imread('itm.tif');
Is=imread('Is.tif');
%Is=rgb2gray(Is);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------create edge and system edge images------------------------------------------------------------------------------------------------------------------------

Iedg=edge(Is,'canny'); % Take canny edge images of Is with automatic threshold
%}
%--------------------------------------------------------------------------------------------------------------------------------------
[y x]=find(Itm>0); % find all y,x cordinates of all points equal 1 inbinary template image Itm
nvs=size(x);% number of points in the  template image
if (nvs<1) disp('error no points find in in template in generalize hought transform, teriminating'); quit() ; end
%-------------------Define Yc and Xc ----------------------------------------------
Cy=1;%round(mean(y));% find object y center, note that any reference point will do so the origin of axis hence 1 could be used just as well
Cx=1;%round(mean(x));% find object z center, note that any reference point will do so the origin of axis hence 1 could be used just as well

%------------------------------create gradient map of Itm, distrobotion between zero to pi %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GradientMap = gradient_direction( Itm );

%%%%%%%%%%%%%%%%%%%%%%%Create an R-Table of Itm gradients to  parameter space in parameter space.%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------create template descriptor array------------------------------------
MaxAngelsBins=30;% devide the angel space to MaxAngelsBins uniformed space bins
MaxPointsPerangel=nvs(1);% maximal amount of points corresponding to specific angel

PointCounter=zeros(MaxAngelsBins);% counter for the amount of edge points associate with each angel gradient
Rtable=zeros(MaxAngelsBins,MaxPointsPerangel,2); % assume maximum of 100 points per angle with MaxAngelsBins angles bins between zero and pi and x,y for the vector to the center of each point
% the third adimension are vector from the point to the center of the vessel

%------------------fill the  angel bins with points in the Rtable---------------------------------------------------------
for f=1:1:nvs(1)
    bin=round((GradientMap(y(f), x(f))/pi)*(MaxAngelsBins-1))+1; % transform from continues gradient angles to discrete angle bins and 
    PointCounter(bin)=PointCounter(bin)+1;% add one to the number of points in the bin
    if (PointCounter(bin)>MaxPointsPerangel)
        disp('exceed max bin in hugh transform');
    end;
    Rtable(bin, PointCounter(bin),1)= Cy-y(f);% add the vector from the point to the object center to the bin
    Rtable(bin, PointCounter(bin),2)= Cx-x(f);% add the vector from the point to the object center to the bin
end;
%plot(pc);
%pause;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%create and populate hough space%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-----------------------------use the array in previous image to identify the template in the main image Is----------------------------------------
[y x]=find(Iedg>0); % find all edg point in the in edge image Iedg of the main image Is
np=size(x);% find number of edge points Is edge image

if (np<1) disp('error no points find in in edge image in generalize hought transform, teriminating'); quit() ; end;

GradientMap=gradient_direction(Is); % create gradient direction  map of the Is
Ss=size(Is); % Size of the main image Is
houghspace=zeros(size(Is));% the hough space assume to be in size of the image but it should probably be smaller
    for f=1:1:np(1)
          bin=round((GradientMap(y(f), x(f))/pi)*(MaxAngelsBins-1))+1; % transform from continues gradient angles to discrete angle bins and 

          for fb=1:1:PointCounter(bin)
              ty=Rtable(bin, fb,1)+ y(f);
              tx=Rtable(bin, fb,2)+ x(f);
               if (ty>0) && (ty<Ss(1)) && (tx>0) && (tx<Ss(2))  
                   houghspace(Rtable(bin, fb,1)+ y(f), Rtable(bin, fb,2)+ x(f))=  houghspace(Rtable(bin, fb,1)+ y(f), Rtable(bin, fb,2)+ x(f))+1; % add point in were the center of the image should be according to the pixel gradient
               end;        
          end;
    end;

%{
%====================================show The Hough Space in color==================================================================================================
imtool(houghspace);
imshow(houghspace,[]);
colormap jet
colorbar
pause
%}

%============================================Find best match in hough space=========================================================================================

%---------------------------------------------------------------------------normalized according to template size (fraction of the template points that was found)------------------------------------------------------------------------------------------------
Itr=houghspace;%./(sum(sum(Itm))); % Itr become the new score matrix
%---------------------------------------------------------------------------find  the location best score all scores which are close enough to the best score
%imtool(Itr,[]);
mx=max(max(Itr));% find the max score location
[y,x]=find(Itr==mx);% 
score=Itr(y,x); % find max score in the huogh space 
%-------------------------------------mark the best result on the system image (Optional Part demand addition function find2 and set2 given below)---------------------------------------------------------------------------
    k =find2(Itm,1);
    mrk=set2(Is,k,255,y(1),x(1)); %paint the templa itm on the image Is
    imshow(mrk);
  %pause();

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ADDITIONAL FUNTIONS ESSENTIAL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%gradient_direction( i3 )%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%gradient_direction( i3 )%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%gradient_direction( i3 )%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%gradient_direction( i3 )%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%gradient_direction( i3 )%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%gradient_direction( i3 )%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [ Is ] = gradient_direction( i3 )
%return map of the absolute direction from -pi/2 to pi/2  of gradient in every point of the gradient  i3(only half circle does not have negative directionss
%-------------------------------------------------------------------
Dy=imfilter(double(i3),[1; -1],'same');%x first derivative  sobel mask
Dx=imfilter(double(i3),[1  -1],'same');% y sobel first derivative
%Is=atan2(Dy,Dx)+pi();
Is=mod(atan2(Dy,Dx)+pi(), pi());%atan(Dy/Dx);%note that this expression can reach infinity if dx is zero mathlab aparently get over it but you can use the folowing expression instead slower but safer: 
%mod(atan2(Dy,Dx)+pi(), pi());%gradient direction map going from 0-180
%--------------------show the image-----------------------------------------------
%{
imshow(Is,[]);% the ,[]  make sure the display will be feeted to doube image
colormap jet
colorbar
pause;
%}
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%FUNCTIONS USED FOR MARKING TEMPLATE ON IMAGE (OPTIONAL)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%set2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%set2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%set2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%set2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%set2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%set2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [m2] = set2( m,k,v,y ,x )
% set value of  v in coordinates k of image m  with initial position x and y,
if nargin<4
    y=0;
    x=0;
end;
if x<0
    x=0;
end;
if y<0
    y=0;
end;
[a,b]=size(k);
m2=m;
for f=1:1:a
    m2(k(f,1)+y,k(f,2)+x)=v;
%m(6,6)=7;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Find2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Find2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Find2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Find2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Find2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Find2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ p ] = find2(m, t )
%FIND2 find points in matrix m their value is above t return them as array
%p
%   Detailed explanation goes here
np=0;
[my,mx]=size(m);
for fy=1:1:my
    for fx=1:1:mx
        if (m(fy,fx)>=t)
           np=np+1;
           p(np,1)=fy;
            p(np,2)=fx;
        end;
    end
    
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


