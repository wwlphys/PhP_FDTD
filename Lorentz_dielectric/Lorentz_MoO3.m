clear
epsfx=4.0;epsfy=5.2;epsfz=2.4;
wLOx=972;wLOy=851;wLOz=1004;
wTOx=820;wTOy=545;wTOz=958;
gx=4;gy=4;gz=2;
fid=fopen('dielectric_MoO3_Lorentz.txt','w');
for w=1:0.3:2000
    epsx=epsfx*(1+(wLOx^2-wTOx^2)/(wTOx^2-w^2-1j*w*gx));
    epsy=epsfy*(1+(wLOy^2-wTOy^2)/(wTOy^2-w^2-1j*w*gy));
    epsz=epsfz*(1+(wLOz^2-wTOz^2)/(wTOz^2-w^2-1j*w*gz));
    fprintf(fid,'%g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g\n',w*0.03e12,0,w,real(epsx),0,0,real(epsy),0,real(epsz),imag(epsx),0,0,imag(epsy),0,imag(epsz));
end
fclose(fid);
