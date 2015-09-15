import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy.coordinates import SkyCoord
from astropy import units as u

#Some handy unit conversions
#days to seconds
daytosec=86400.0
#AU to meters
autom=149597870700.0
#pc to meters
pctom=3.08567758e16
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
    r=a*(1-ecc*np.cos(e))#physical distance between star and planet
    xref=r*(np.cos(om)*np.cos(w+f)-np.sin(om)*np.sin(w+f)*np.cos(i))#X (x-coordinate of planet in observer's frame)
    yref=r*(np.sin(om)*np.cos(w+f)+np.cos(om)*np.sin(w+f)*np.cos(i))#Y (y-coordinate of planet in observer's frame)
    zref=r*np.sin(w+f)*np.sin(i)#Z (z-coordinate of planet in observer's frame)
    rpro=(xref**2+yref**2)**0.5#projected separation onto sky
    posang=pa(xref,yref)#Position angle of planet with respect to reference direction (N)
    rs=m2*r/(m1+m2)#Physical distance of star from center of mass
    rp=m1*r/(m1+m2)#Physical distance of planet from center of mass
    xs=rs*(np.cos(om)*np.cos(w+f)-np.sin(om)*np.sin(w+f)*np.cos(i))#x-coordinate of star wrt center of mass
    ys=rs*(np.sin(om)*np.cos(w+f)+np.cos(om)*np.sin(w+f)*np.cos(i))#y-coordinate of star wrt center of mass
    zs=rs*np.sin(w+f)*np.sin(i)#y-coordinate of star wrt center of mass
    xp=-rp*(np.cos(om)*np.cos(w+f)-np.sin(om)*np.sin(w+f)*np.cos(i))#x-coordinate of planet wrt center of mass
    yp=-rp*(np.sin(om)*np.cos(w+f)+np.cos(om)*np.sin(w+f)*np.cos(i))#y-coordinate of planet wrt center of mass
    zp=-rp*np.sin(w+f)*np.sin(i)#z-coordinate of planet wrt center of mass
    rspro=(xs**2+ys**2)**0.5#projected separation onto sky of star from COM
    rppro=(xp**2+yp**2)**0.5#projected separation onto sky of planet from COM
    pas=pa(xs,ys)#position angle of star wrt COM
    pap=pa(xp,yp)#position angle of planet wrt COM
    #Velocity semi-amplitude (m/s) and radial velocities (m/s) of star and planet wrt COM
    ks=2*np.pi*m2*a*np.sin(i)/(86400.0*p*(m1+m2)*(1-ecc**2)**0.5)
    kp=2*np.pi*m1*a*np.sin(i)/(86400.0*p*(m1+m2)*(1-ecc**2)**0.5)
    rvs=ks*(np.cos(w+f)+ecc*np.cos(w))
    rvp=kp*(np.cos(w+f)+ecc*np.cos(w))
    return {'proj_sep':rpro, 'p_pa':posang, 's_pacom':pas, 'p_pacom':pap, \
    's_rv':rvs, 'p_rv':rvp, 'xs':xs, 'ys':ys, 'zs':zs, \
    'xp':xp, 'yp':yp, 's_proj_sep':rspro}

#Obliquity of the ecliptic: 23:27:08.26 - 0.4684"*(t - 2415020.31352)/365.25; perhaps just use 23.45 for simplicity...
def obliquity(jd):
    ob_deg=23.452294444444-0.4684*(jd-2415020.31352)/(365.25*3600.0)
    ob_rad=ob_deg*degtorad
    return ob_rad

#Ecliptic Longitude of the sun for a given JD
def lambdasun(jd):
    d=jd-2451545.0
    g=357.528+0.9856003*d
    l=280.46+0.9856474*d
    lamb_deg=l+1.915*np.sin(g)+.020*np.sin(2*g)
    lamb_rad=lamb_deg*degtorad
    return lamb_rad

#Convert RA and Dec to Ecliptic coordinates
def equattoeclip(ra,dec,ob):
    beta=np.arcsin(np.sin(dec)*np.cos(ob)-np.cos(dec)*np.sin(ob)*np.sin(ra))
    lamb=np.arccos(np.cos(ra)*np.cos(dec)/np.cos(beta))
    return {'lambda':lamb, 'beta':beta}

#Convert Ecliptic to RA and Dec coordinates
def ecliptoequat(lamb,beta,ob):
    delta=np.arcsin(np.sin(beta)*np.cos(ob)+np.cos(beta)*np.sin(ob)*np.sin(lamb))
    alpha=np.arccos(np.cos(lamb)*np.cos(beta)/np.cos(delta))
    return {'ra':alpha, 'dec':delta}

#From Simbad, position of HD 80606 (assuming it represents COM) was (09h22m37.5797s, +50d36m13.4818s) J2000
equat_position=SkyCoord('09h22m37.5797s', '+50d36m13.4818s',frame='icrs')
ra2000_deg=equat_position.ra.degree
dec2000_deg=equat_position.dec.degree
ra2000_rad=equat_position.ra.radian
dec2000_rad=equat_position.dec.radian

#Obliquity for J2000 was then...
ob2000=obliquity(2451545.0)

#Ecliptic coordinates of HD 80606 were then...
eclip_position=equattoeclip(ra2000_rad,dec2000_rad,ob2000)
lam2000_rad=eclip_position['lambda']
beta2000_rad=eclip_position['beta']

#Ecliptic longitude of the Sun for J2000 was then...
ecsunlam2000=lambdasun(2451545.0)

del_lam2000=5.63*np.sin(ecsunlam2000-lam2000_rad)*mastodeg*degtorad
del_beta2000=5.63*np.cos(ecsunlam2000-lam2000_rad)*np.sin(beta2000_rad)*mastodeg*degtorad

new_equat_position=ecliptoequat(lam2000_rad+del_lam2000,beta2000_rad+del_beta2000,ob2000)
new_ra2000=new_equat_position['ra']
new_dec2000=new_equat_position['dec']

#HD 80606 system orbital elements - Hebrard et al. 2010 (H10); Wikitorowicz & Laughlin 2014
#p = 111.4367 days    (H10)
#t0 = 2455204.916 HJD (H10)
#ecc = 0.933          (H10)
#a = 0.455 au         (H10)
#i = 89.269 degrees   (H10)
#om = 160.98 degrees  (W14)
#w = 300.77 degrees   (H10)
#m1 = 1.01 msun       (H10)
#m2 = 4.08 mjup       (H10)
#r1 = 1.007 Rsun
#r2 = 0.981 Rjup
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
zstar=np.zeros(len(fall))
for i0 in range(len(fall)):
    hjd[i0]=fall[i0]+2457235.49510
    result=orb_pred(a,ecc,i,om1,w,t0,hjd[i0],p,m1,m2)
    rv[i0]=result['s_rv']
    rpro[i0]=result['proj_sep']
    zstar[i0]=result['zs']

rvmax=np.argmax(rv)
rvmin=np.argmin(rv)
print hjd[rvmin],hjd[rvmax]#2457320.8061 2457322.4831; Oct 25 and 27

rpro_tran=1.007*rsun+0.981*rjup

transit=np.where((np.abs(rpro) < rpro_tran) & (zstar > 0.0))
firstcontact=min(hjd[transit])
fourthcontact=max(hjd[transit])
print firstcontact,fourthcontact#Nov 1 02:19:36 UTC Nov 1 14:13:46 UTC
#Oct 31 20/1?:19:36 CST Nov 1 08:13:46 CST
#After full moon; Sun sets at 18:18; Sun rises at 07:02 - HD 80606 airmass < 2 around 3am :(

#plt.plot(hjd-2.4572e6,rv)
#plt.plot(hjd[transit]-2.4572e6,rv[transit],'r')
#plt.xlim((2457235-2.4572e6,2457389-2.4572e6))
#plt.xlabel('HJD-2457200')
#plt.ylabel('Radial Velocity (m/s)')
#plt.title('HD 80606 Radial Velocity Curve')
#plt.savefig("/Users/ram/Dropbox/ut/2015fa/ast381/hw1/rv.eps")

#PM=(45.76, 16.56) mas/yr
#PLX=5.63 mas
#Jan 1, 2014 to Jan 1, 2019 (2456658.5,2458484.5)

gaia=np.arange(0,182600,dtype=float)/100.0
#Planets influence on the star
ra_p=np.zeros(len(gaia))
dec_p=np.zeros(len(gaia))
for i1 in range(len(gaia)):
    temp=orb_pred(a,ecc,i,om1,w,t0,gaia[i1]+2456658.5,p,m1,m2)
    del_dec=temp['s_proj_sep']*np.cos(temp['s_pacom'])*5.63e-3/(pctom*degtorad)
    del_ra=temp['s_proj_sep']*np.sin(temp['s_pacom'])*5.63e-3/(pctom*degtorad)
    ra_p[i1]=del_ra
    dec_p[i1]=del_dec

#plt.plot(dec_p,ra_p)
#plt.show()

ra_pm=np.zeros(len(gaia))
dec_pm=np.zeros(len(gaia))
ra_plx=np.zeros(len(gaia))
dec_plx=np.zeros(len(gaia))
for i1 in range(len(gaia)):
    ra_pm[i1]=ra2000_deg+(gaia[i1]+2456658.5-2451545.0)*45.78*mastodeg/365.25
    dec_pm[i1]=dec2000_deg+(gaia[i1]+2456658.5-2451545.0)*16.56*mastodeg/365.25

#plt.plot(dec_pm-dec2000_deg,ra_pm-ra2000_deg)
#plt.xlim(max(dec_pm)-dec2000_deg,min(dec_pm)-dec2000_deg)
#plt.ylim(max(ra_pm)-ra2000_deg,min(ra_pm)-ra2000_deg)
#plt.show()

    
