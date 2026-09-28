clear
wp=2002.6e12;
g =11.61e12;
wn(1)=197.3e12;
wn(2)=1083.5e12;
wn(3)=1979.1e12;
wn(4)=4932.5e12;
wn(5)=9812.1e12;
gn(1)=939.62e12;
gn(2)=109.29e12;
gn(3)=15.71e12;
gn(4)=221.49e12;
gn(5)=584.91e12;
wp=wp/0.03e12;
g=g/0.03e12;
wn=wn/0.03e12;
gn=gn/0.03e12;
fn(1)=7.9247;
fn(2)=0.5013;
fn(3)=0.0133;
fn(4)=0.8266;
fn(5)=1.1133;
fid=fopen('dielectric_Ag_Lorentz.txt','w');
for w=1:0.3:2000
    eps=1-wp^2/w/(w-1j*g);
    for i = 1:5
        eps = eps + fn(i)*wn(i)^2/(wn(i)^2-w^2+1j*w*gn(i));
    end
    fprintf(fid,'%g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g\n',w*0.03e12,0,w,real(eps),0,0,real(eps),0,real(eps),-imag(eps),0,0,-imag(eps),0,-imag(eps));
end
fclose(fid);
