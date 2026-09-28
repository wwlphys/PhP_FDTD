clear
wr = 215; % cm-1
wp = 72800; % cm-1
fid=fopen('dielectric_Au_Lorentz.txt','w');
for w=1:0.3:2000
    eps1 = -wp^2/(w^2+wr^2);
    eps2 = wr*wp^2/w/(w^2+wr^2);
    fprintf(fid,'%g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g\n',w*0.03e12,0,w,real(eps1),0,0,real(eps1),0,real(eps1),eps2,0,0,eps2,0,eps2);
end
fclose(fid);
