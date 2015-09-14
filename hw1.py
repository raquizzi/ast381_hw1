import numpy as np
import matplotlib.pyplot as plt
import scipy
import astropy.coordinates as coord
import astropy.units as u

#Some handy unit conversions
#days to seconds
daytosec=86400.0
#AU to meters
autom=149597870700.0
#Degrees to radians
degtorad=np.pi/180.0
#Mass of the sun in kg
msun=1.989e30
#Mass of Jupiter in kg
mjup=1.898e27
#Radius of Sun to meters
rsun=6.963e8
#Radius of Jupiter to meters
rjup=6.9911e7
#mas to degrees
mastodeg=1/(3.6e6)

#kep_nr is my own Newton Raphson Solver
#Iterates until g is within 1e-6 of 0
def kep_nr(ecc,m):
    e_int=m
    g_int=e_int-ecc*np.sin(e_int)-m
    dg_int=1-ecc*np.cos(e_int)
    e_old=e_int
    g=g_int
    dg=dg_int
    abovethresh=True
    while abovethresh:
        e_new=e_old-g/dg
        e_old=e_new
        dg=1-ecc*np.cos(e_new)
        g=e_new-ecc*np.sin(e_new)-m
        if np.abs(g) < 1e-6:
            abovethresh=False
    return e_new

#pa is a function that returns the postion angle based on the X and Y (observer's) coordinates of the planet
#+X=N; +Y=W; +Z=Away from observer
def pa(x,y):
    r=(x**2.0+y**2.0)**0.5
    if y <= 0.0:
        posang=np.arccos(x/r)
    elif x < 0.0 and y > 0.0:
        posang=np.arctan(y/x)+np.pi
    elif x >= 0.0 and y > 0.0:
        posang=np.arcsin(y/r)+2.0*np.pi
    return posang

#Now for the orbit prediction code!
#a=semi-major axis (m)
#ecc=eccentricity
#i=inclination (radians)
#om=capital omega (radians)
#w=lowercase omega (radians)
#t0=time of periastron (HJD)
#t=time of observation (HJD)
#p=period (days)
#m1=mass of primary (kg)
#m2=mass of secondary (kg)
def orb_pred(a,ecc,i,om,w,t0,t,p,m1,m2):
    m=2*np.pi*(t-t0)/p#mean anomaly
    e=kep_nr(ecc,m)#eccentric anomaly
    tanf2=np.tan(e/2)*((1+ecc)/(1-ecc))**0.5#tan(true anomaly/2)
    f=2*np.arctan(tanf2)#true anomaly
    r=a*(1-ecc*np.cos(e))#distance between star and planet
    #xorb=r*np.cos(f)
    #yorb=r*np.sin(f)
    #zorb=0
    xref=r*(np.cos(om)*np.cos(w+f)-np.sin(om)*np.sin(w+f)*np.cos(i))#X (x-coordinate in observer's frame)
    yref=r*(np.sin(om)*np.cos(w+f)+np.cos(om)*np.sin(w+f)*np.cos(i))#Y (y-coordinate in observer's frame)
    zref=r*np.sin(w+f)*np.sin(i)#Z (z-coordinate in observer's frame)
    rpro=(xref**2+yref**2)**0.5#projected separation in observer's frame
    #print "On-sky projected separation between the objects is "+str(rpro)+"!"
    posang=pa(xref,yref)#Position angle with respect to reference direction (N)
    #print "Position angle of B with respect to A is "+str(posang)+"!"
    rcom1=m2*r/(m1+m2)#Center of Mass r1 vector length
    rcom2=m1*r/(m1+m2)#Center of Mass r2 vector length
    #Coordinates of star and planet in COM frame
    xcomref1=rcom1*(np.cos(om)*np.cos(w+f)-np.sin(om)*np.sin(w+f)*np.cos(i))
    ycomref1=rcom1*(np.sin(om)*np.cos(w+f)+np.cos(om)*np.sin(w+f)*np.cos(i))
    zcomref1=rcom1*np.sin(w+f)*np.sin(i)
    xcomref2=rcom2*(np.cos(om)*np.cos(w+f)-np.sin(om)*np.sin(w+f)*np.cos(i))
    ycomref2=rcom2*(np.sin(om)*np.cos(w+f)+np.cos(om)*np.sin(w+f)*np.cos(i))
    zcomref2=rcom2*np.sin(w+f)*np.sin(i)
    rcompro1=(xcomref1**2+ycomref1**2)**0.5
    rcompro2=(xcomref2**2+ycomref2**2)**0.5
    pacom1=pa(xcomref1,ycomref1)
    pacom2=pa(xcomref2,ycomref2)
    #print "Position angle of A with respect to the COM is "+str(pacom1)+"!"
    #print "Position angle of B with respect to the COM is "+str(pacom2)+"!"
    #Velocity semi-amplitude (m/s) and radial velocities (m/s) of star and planet
    k1=2*np.pi*m2*a*np.sin(i)/(86400.0*p*(m1+m2)*(1-ecc**2)**0.5)
    k2=2*np.pi*m1*a*np.sin(i)/(86400.0*p*(m1+m2)*(1-ecc**2)**0.5)
    rv1=k1*(np.cos(w+f)+ecc*np.cos(w))
    rv2=k2*(np.cos(w+f)+ecc*np.cos(w))
    #print "The RV of A with respect to the COM is "+str(rv1)+"!"
    #print "The RV of B with respect to the COM is "+str(rv2)+"!"
    return {'proj_sep':rpro, 'PA':posang, 'PAcom_s':pacom1, 'PAcom_p':pacom2, 'RVcom_s':rv1, 'RVcom_p':rv2, 'xs':xcomref1, 'ys':ycomref1, 'xp':xcomref2, 'yp':ycomref2}

#HD 80606 system orbital elements
#p=111.4367 days
#t0=2455204.916 HJD
#ecc=0.934
#a=0.455 AU
#i=89.269 degrees
#om=-19.02 or 160.98 degrees
#w=300.77 degrees
#m1=1.01 msun
#m2=4.08 mjup
#HJD on Aug 1, 2015 = 2457235.49510
#HJD on Jan 1, 2016 = 2457388.50426

a=0.455*autom
ecc=0.934
i=89.269*degtorad
om1=160.98*degtorad
om2=-19.02*degtorad
w=300.77*degtorad
t0=2455204.916
p=111.4367#*daytosec
m1=1.01*msun
m2=4.08*mjup

fall=np.arange(0,154000,dtype=float)/1000.0
rv=np.zeros(len(fall))
rpro=np.zeros(len(fall))
hjd=np.zeros(len(fall))
for i0 in range(len(fall)):
    hjd[i0]=fall[i0]+2457235.49510
    result=orb_pred(a,ecc,i,om1,w,t0,hjd[i0],p,m1,m2)
    rv[i0]=result['RVcom_s']
    rpro[i0]=result['proj_sep']

rvmax=np.argmax(rv)
rvmin=np.argmin(rv)
print hjd[rvmin],hjd[rvmax]#2457320.8061 2457322.4831; Oct 25 and 27

#Radius of HD 80606 = 0.98 Rsun
#Radius of HD 80606b= 0.921 Rjup
rpro_tran=0.98*rsun+0.921*rjup

transit=np.where((np.abs(rpro) < rpro_tran) & (hjd > 2457322.4831))
firstcontact=min(hjd[transit])
fourthcontact=max(hjd[transit])
print firstcontact,fourthcontact#Nov 1 03:32:59 UTC Nov 1 13:00:20 UTC
#Oct 31 21:32:59 CST Nov 1 07:00:20 CST
#After full moon; Sun sets at 18:18; gain an hour

plt.plot(hjd-2.4572e6,rv)
plt.plot(hjd[transit]-2.4572e6,rv[transit],'r')
plt.xlim((2457235-2.4572e6,2457389-2.4572e6))
plt.xlabel('HJD+2457200')
plt.ylabel('Radial Velocity (m/s)')
plt.title('HD 80606 Radial Velocity Curve')
plt.savefig("rv.eps")

#Obliquity of the ecliptic: 23:27:08.26 - 0.4684*(t - 1900); perhaps just use 23.45 for simplicity...
def equattoeclip(ra,dec,ob):
    beta=np.arcsin(np.sin(dec)*np.cos(ob)-np.cos(dec)*np.sin(ob)*np.sin(ra))
    lamb=np.arccos(np.cos(ra)*np.cos(dec)/np.cos(beta))
    return {'lambda':lamb, 'beta':beta}
#Longitude of the sun
def lambdasun(date):
    d=date-2447162.00428
    e=0.016714
    g=-0.04144+0.0172019696*d
    l=-1.38691+0.017202124*d
    lamb=l+2*e*np.sin(g)+1.2*(e**2)*np.sin(2*g)
    return lamb
#Position of HD 80606 on HJD 2448349.06443 was
#RA=140.65638797
#Dec=+50.60370469
#PM=(45.76,16.56) mas/yr
#PLX=5.63 mas
#Jan 1, 2014 to Jan 1, 2019 (2456658.50428,2458484.50427)

gaia=np.arange(0,182600,dtype=float)/100.0
ra2014=140.65638797+(5.63*np.sin((2448349.06443-2456658.50428)/365.0)/3.6e6)+(2448349.06443-2456658.50428)*((45.76**2+16.56**2)**0.5)*(np.cos(np.arctan(16.56/45.76)))/(3.6e6*365.0)
dec2014=50.60370469+(2448349.06443-2456658.50428)*((45.76**2+16.56**2)**0.5)*(np.sin(np.arctan(16.56/45.76)))/(3.6e6*365.0)

ra2019=140.65638797+(5.63*np.sin((2456658.50428-2458484.50427)/365.0)/3.6e6)+(2456658.50428-2458484.50427)*((45.76**2+16.56**2)**0.5)*(np.cos(np.arctan(16.56/45.76)))/(3.6e6*365.0)
dec2019=50.60370469+(2456658.50428-2458484.50427)*((45.76**2+16.56**2)**0.5)*(np.sin(np.arctan(16.56/45.76)))/(3.6e6*365.0)
    
