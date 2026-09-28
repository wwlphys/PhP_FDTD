amplitude=1.0
wavenumber=500
frequency=wavenumber*0.03e12
t0=0.2e-12
tau=0.075e-12
time = 0:1e-15:500e-15;
s = amplitude* sin(2 * pi * frequency * time) .* exp(-((time - t0) / tau).^2);
plot(time,s,'*')