clear
eps0=1.0;
s1=184.1414; w1=696.0241; g1=15.3851;
s2=126.0974; w2=758.5979; g2=19.8793;
fid=fopen('dielectric_ARP_Lorentz.txt','w');
for w=1:0.3:2000
    eps=eps0+s1^2/(w1^2-w^2-1j*w*g1)+s2^2/(w2^2-w^2-1j*w*g2);
    fprintf(fid,'%g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g\n',w*0.03e12,0,w,real(eps),0,0,real(eps),0,real(eps),imag(eps),0,0,imag(eps),0,imag(eps));
end
fclose(fid);
