function y=U_thres(y,up,low)
y(y>up) = up;
y(y<low) = low;
