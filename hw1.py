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
        posang=np.arctan(-y/x)+np.pi
    elif x >= 0.0 and y > 0.0:
        posang=np.arcsin(-y/r)+2.0*np.pi
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

#This stuff was really for #4, but I wanted to define them up with the other independent functions
#Also did some preliminary comparisons with the NED calculator with print statements, but deleted them once I knew the codes were doing the right thing
#Obliquity of the ecliptic: 23:27:08.26 - 0.4684"*(t - 2415020.31352)/365.25
def obliquity(jd):
    ob_deg=23.452294444444-0.4684*(jd-2415020.31352)/(365.25*3600.0)
    ob_rad=ob_deg*degtorad
    return ob_rad

#Ecliptic Longitude of the sun for a given JD
def lambdasun(jd):
    d=jd-2451545.0
    g=357.528+0.9856003*d*degtorad
    l=280.46+0.9856474*d
    lamb_deg=l+1.915*np.sin(g)+0.020*np.sin(2*g)
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

#Parallaxes (mas): Hipparcos 2007; Hipparcos 1997
plx1=5.63
plx2=17.13
del_lam2000_1=plx1*np.sin(ecsunlam2000-lam2000_rad)*mastodeg*degtorad/np.cos(beta2000_rad)
del_beta2000_1=plx1*np.cos(ecsunlam2000-lam2000_rad)*np.sin(beta2000_rad)*mastodeg*degtorad
del_lam2000_2=plx2*np.sin(ecsunlam2000-lam2000_rad)*mastodeg*degtorad/np.cos(beta2000_rad)
del_beta2000_2=plx2*np.cos(ecsunlam2000-lam2000_rad)*np.sin(beta2000_rad)*mastodeg*degtorad

new_equat_position1=ecliptoequat(lam2000_rad+del_lam2000_1,beta2000_rad+del_beta2000_1,ob2000)
new_ra2000_1=new_equat_position1['ra']/degtorad
new_dec2000_1=new_equat_position1['dec']/degtorad
new_equat_position2=ecliptoequat(lam2000_rad+del_lam2000_2,beta2000_rad+del_beta2000_2,ob2000)
new_ra2000_2=new_equat_position2['ra']/degtorad
new_dec2000_2=new_equat_position2['dec']/degtorad

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
#r1 = 1.007 rsun      (H10)
#r2 = 0.981 Rjup      (H10)
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
#print hjd[rvmin],hjd[rvmax]#2457320.8061 2457322.4831; Oct 25 and 27

#2457320.8061 2457322.4831
#2457327.5971 2457328.0931 - 1st and 4th contact

rpro_tran=1.007*rsun+0.981*rjup

transit=np.where((np.abs(rpro) < rpro_tran) & (zstar > 0.0))
firstcontact=min(hjd[transit])
fourthcontact=max(hjd[transit])
#print firstcontact,fourthcontact#Nov 1 02:19:36 UTC Nov 1 14:13:46 UTC
#Oct 31 20/1?:19:36 CST Nov 1 08:13:46 CST
#After full moon; Sun sets at 18:18; Sun rises at 07:02 - HD 80606 airmass < 2 around 3am :(

plt.figure(0)
plt.plot(hjd-2.4572e6,rv,linewidth=4)
plt.plot(hjd[transit]-2.4572e6,rv[transit],'r',linewidth=4)
plt.xlim((2457235-2.4572e6,2457389-2.4572e6))
plt.xlabel('HJD-2457200')
plt.ylabel('Radial Velocity (m/s)')
plt.title('HD 80606 Radial Velocity Curve')
plt.savefig("/Users/ram/Dropbox/ut/2015fa/ast381/hw1/rv.eps")
#plt.savefig("/Users/rmartinez/Dropbox/ut/2015fa/ast381/hw1/rv.eps")

#PM=(45.76, 16.56) mas/yr
#PLX=5.63 mas OR 17.13 from Hipparcos 1997 that people still cite...
#Jan 1, 2014 to Jan 1, 2019 (2456658.5,2458484.5)

gaia=np.arange(0,182600,dtype=float)/100.0
#Planet's influence on the star!
ra_p2=np.zeros(len(gaia))
dec_p2=np.zeros(len(gaia))
ra_p1=np.zeros(len(gaia))
dec_p1=np.zeros(len(gaia))
plx1=5.63
plx2=17.13
for i1 in range(len(gaia)):
    temp=orb_pred(a,ecc,i,om1,w,t0,gaia[i1]+2456658.5,p,m1,m2)
    del_dec1=temp['s_proj_sep']*np.cos(temp['s_pacom'])*plx1*1e-3/(pctom*degtorad)
    del_ra1=temp['s_proj_sep']*np.sin(temp['s_pacom'])*plx1*1e-3/(pctom*degtorad)
    del_dec2=temp['s_proj_sep']*np.cos(temp['s_pacom'])*plx2*1e-3/(pctom*degtorad)
    del_ra2=temp['s_proj_sep']*np.sin(temp['s_pacom'])*plx2*1e-3/(pctom*degtorad)
    ra_p1[i1]=del_ra1*3.6e9
    dec_p1[i1]=del_dec1*3.6e9
    ra_p2[i1]=del_ra2*3.6e9
    dec_p2[i1]=del_dec2*3.6e9

#Add parallax! Going to assume RA, Dec of HD 80606 is stationary at its J2000 position for simplicity...
ra_plx1=np.zeros(len(gaia))
dec_plx1=np.zeros(len(gaia))
ra_plx2=np.zeros(len(gaia))
dec_plx2=np.zeros(len(gaia))
for i2 in range(len(gaia)):
    ob=obliquity(gaia[i2]+2456658.5)
    lamsun=lambdasun(gaia[i2]+2456658.5)
    eclip_position=equattoeclip(ra2000_rad,dec2000_rad,ob)
    lam_rad=eclip_position['lambda']
    beta_rad=eclip_position['beta']
    del_lam1=plx1*np.sin(lamsun-lam_rad)*mastodeg*degtorad/np.cos(beta_rad)
    del_beta1=plx1*np.cos(lamsun-lam_rad)*np.sin(beta_rad)*mastodeg*degtorad
    del_lam2=plx2*np.sin(lamsun-lam_rad)*mastodeg*degtorad/np.cos(beta_rad)
    del_beta2=plx2*np.cos(lamsun-lam_rad)*np.sin(beta_rad)*mastodeg*degtorad
    new_equat_position1=ecliptoequat(lam_rad+del_lam1,beta_rad+del_beta1,ob)
    new_equat_position2=ecliptoequat(lam_rad+del_lam2,beta_rad+del_beta2,ob)
    ra_plx1[i2]=(new_equat_position1['ra']/degtorad-ra2000_deg)*1e3/mastodeg+ra_p1[i2]
    dec_plx1[i2]=(new_equat_position1['dec']/degtorad-dec2000_deg)*1e3/mastodeg+dec_p1[i2]
    ra_plx2[i2]=(new_equat_position2['ra']/degtorad-ra2000_deg)*1e3/mastodeg+ra_p2[i2]
    dec_plx2[i2]=(new_equat_position2['dec']/degtorad-dec2000_deg)*1e3/mastodeg+dec_p2[i2]

#Now add proper motion!
ra_pm1=np.zeros(len(gaia))
dec_pm1=np.zeros(len(gaia))
ra_pm2=np.zeros(len(gaia))
dec_pm2=np.zeros(len(gaia))
for i3 in range(len(gaia)):
    ra_pm1[i3]=(gaia[i3]+2456658.5-2451545.0)*45.78*1e3/365.25+ra_plx1[i3]
    dec_pm1[i3]=(gaia[i3]+2456658.5-2451545.0)*16.56*1e3/365.25+dec_plx1[i3]
    ra_pm2[i3]=(gaia[i3]+2456658.5-2451545.0)*45.78*1e3/365.25+ra_plx2[i3]
    dec_pm2[i3]=(gaia[i3]+2456658.5-2451545.0)*16.56*1e3/365.25+dec_plx2[i3]

#Time to fake some data!
N=100
g_sim=np.sort(1826.0*np.random.rand(N)+2456658.5)
ra_p_sim1=np.zeros(len(g_sim))
dec_p_sim1=np.zeros(len(g_sim))
ra_plx_sim1=np.zeros(len(g_sim))
dec_plx_sim1=np.zeros(len(g_sim))
ra_pm_sim1=np.zeros(len(g_sim))
dec_pm_sim1=np.zeros(len(g_sim))
ra_p_sim2=np.zeros(len(g_sim))
dec_p_sim2=np.zeros(len(g_sim))
ra_plx_sim2=np.zeros(len(g_sim))
dec_plx_sim2=np.zeros(len(g_sim))
ra_pm_sim2=np.zeros(len(g_sim))
dec_pm_sim2=np.zeros(len(g_sim))
#Error arrays
ra_p_simerr=3.0+0.5*np.random.rand(N)
dec_p_simerr=3.0+0.5*np.random.rand(N)
for i4 in range(len(g_sim)):
    t_sim=orb_pred(a,ecc,i,om1,w,t0,g_sim[i4],p,m1,m2)
    del_dec1=t_sim['s_proj_sep']*np.cos(t_sim['s_pacom'])*plx1*1e-3/(pctom*degtorad)
    del_ra1=t_sim['s_proj_sep']*np.sin(t_sim['s_pacom'])*plx1*1e-3/(pctom*degtorad)
    del_dec2=t_sim['s_proj_sep']*np.cos(t_sim['s_pacom'])*plx2*1e-3/(pctom*degtorad)
    del_ra2=t_sim['s_proj_sep']*np.sin(t_sim['s_pacom'])*plx2*1e-3/(pctom*degtorad)
    ra_p_sim_temp1=del_ra1*3.6e9
    dec_p_sim_temp1=del_dec1*3.6e9
    ra_p_sim1[i4]=ra_p_sim_temp1+0.2*np.random.randn()
    dec_p_sim1[i4]=dec_p_sim_temp1+0.2*np.random.randn()
    ra_p_sim_temp2=del_ra2*3.6e9
    dec_p_sim_temp2=del_dec2*3.6e9
    ra_p_sim2[i4]=ra_p_sim_temp2+0.2*np.random.randn()
    dec_p_sim2[i4]=dec_p_sim_temp2+0.2*np.random.randn()
    ob=obliquity(g_sim[i4])
    lamsun=lambdasun(g_sim[i4])
    eclip_position=equattoeclip(ra2000_rad,dec2000_rad,ob)
    lam_rad=eclip_position['lambda']
    beta_rad=eclip_position['beta']
    del_lam1=plx1*np.sin(lamsun-lam_rad)*mastodeg*degtorad/np.cos(beta_rad)
    del_beta1=plx1*np.cos(lamsun-lam_rad)*np.sin(beta_rad)*mastodeg*degtorad
    del_lam2=plx2*np.sin(lamsun-lam_rad)*mastodeg*degtorad/np.cos(beta_rad)
    del_beta2=plx2*np.cos(lamsun-lam_rad)*np.sin(beta_rad)*mastodeg*degtorad
    new_equat_position1=ecliptoequat(lam_rad+del_lam1,beta_rad+del_beta1,ob)
    new_equat_position2=ecliptoequat(lam_rad+del_lam2,beta_rad+del_beta2,ob)
    ra_plx_sim1[i4]=(new_equat_position1['ra']/degtorad-ra2000_deg)*1e3/mastodeg+ra_p_sim1[i4]
    dec_plx_sim1[i4]=(new_equat_position1['dec']/degtorad-dec2000_deg)*1e3/mastodeg+dec_p_sim1[i4]
    ra_pm_sim1[i4]=(g_sim[i4]-2451545.0)*45.78*1e3/365.25+ra_plx_sim1[i4]
    dec_pm_sim1[i4]=(g_sim[i4]-2451545.0)*16.56*1e3/365.25+dec_plx_sim1[i4]
    ra_plx_sim2[i4]=(new_equat_position2['ra']/degtorad-ra2000_deg)*1e3/mastodeg+ra_p_sim2[i4]
    dec_plx_sim2[i4]=(new_equat_position2['dec']/degtorad-dec2000_deg)*1e3/mastodeg+dec_p_sim2[i4]
    ra_pm_sim2[i4]=(g_sim[i4]-2451545.0)*45.78*1e3/365.25+ra_plx_sim2[i4]
    dec_pm_sim2[i4]=(g_sim[i4]-2451545.0)*16.56*1e3/365.25+dec_plx_sim2[i4]

plt.figure(1)
plt.plot(ra_p1,dec_p1,linewidth=2)
plt.errorbar(ra_p_sim1,dec_p_sim1,xerr=ra_p_simerr,yerr=dec_p_simerr,fmt='o')
plt.xlim(max(ra_p1)+5.0,min(ra_p1)-5.0)
plt.ylim(min(dec_p1)-5.0,max(dec_p1)+5.0)
plt.ylabel(r'$\Delta$Dec ($\mu$as)')
plt.xlabel(r'$\Delta$RA ($\mu$as)')
plt.title(r'$\Delta$RA versus $\Delta$Dec')
#plt.savefig("/Users/rmartinez/Dropbox/ut/2015fa/ast381/hw1/dec_v_ra1_plx1.eps")
plt.savefig("/Users/ram/Dropbox/ut/2015fa/ast381/hw1/dec_v_ra1_plx1.eps")

plt.figure(2)
plt.plot(gaia,ra_p1,linewidth=2)
plt.errorbar(g_sim-2456658.5,ra_p_sim1,yerr=ra_p_simerr,fmt='o')
plt.xlabel('JD - 2458484.5')
plt.ylabel(r'$\Delta$RA ($\mu$as)')
plt.xlim(0.0,1826.0)
plt.ylim(-4.0,+8.0)
plt.title(r'$\Delta$RA versus Time')
#plt.savefig("/Users/rmartinez/Dropbox/ut/2015fa/ast381/hw1/ra_v_t1_plx1.eps")
plt.savefig("/Users/ram/Dropbox/ut/2015fa/ast381/hw1/ra_v_t1_plx1.eps")

plt.figure(3)
plt.plot(gaia,dec_p1,linewidth=4)
plt.errorbar(g_sim-2456658.5,dec_p_sim1,yerr=dec_p_simerr,fmt='o')
plt.xlabel('JD-2458484.5')
plt.ylabel(r'Dec ($\mu$as)')
plt.xlim(0.0,1826.0)
plt.ylim(min(dec_p1)-5.0,max(dec_p1)+5.0)
plt.title(r'$\Delta$Dec versus Time')
plt.savefig("/Users/ram/Dropbox/ut/2015fa/ast381/hw1/dec_v_t1_plx1.eps")
#plt.savefig("/Users/rmartinez/Dropbox/ut/2015fa/ast381/hw1/dec_v_t1_plx1.eps")

plt.figure(4)
plt.plot(ra_plx1*1e-3,dec_plx1*1e-3,linewidth=2)
plt.errorbar(ra_plx_sim1*1e-3,dec_plx_sim1*1e-3,xerr=ra_p_simerr*1e-3,yerr=dec_p_simerr*1e-3,fmt='o')
plt.xlim(max(ra_plx1*1e-3)+1.0,min(ra_plx1*1e-3)-1.0)
plt.ylim(min(dec_plx1*1e-3)-1.0,max(dec_plx1*1e-3)+1.0)
plt.ylabel(r'$\Delta$Dec (mas)')
plt.xlabel(r'$\Delta$RA (mas)')
plt.title(r'$\Delta$RA versus $\Delta$Dec')
plt.savefig("/Users/ram/Dropbox/ut/2015fa/ast381/hw1/dec_v_ra2_plx1.eps")
#plt.savefig("/Users/rmartinez/Dropbox/ut/2015fa/ast381/hw1/dec_v_ra2_plx1.eps")

plt.figure(5)
plt.plot(gaia,ra_plx1*1e-3,linewidth=2)
plt.errorbar(g_sim-2456658.5,ra_plx_sim1*1e-3,yerr=ra_p_simerr*1e-3,fmt='o')
plt.xlabel('JD - 2458484.5')
plt.ylabel(r'$\Delta$RA (mas)')
plt.xlim(0.0,1826.0)
plt.ylim(min(ra_plx1*1e-3)-1.0,max(ra_plx1*1e-3)+1.0)
plt.title(r'$\Delta$RA versus Time')
plt.savefig("/Users/ram/Dropbox/ut/2015fa/ast381/hw1/ra_v_t2_plx1.eps")
#plt.savefig("/Users/rmartinez/Dropbox/ut/2015fa/ast381/hw1/ra_v_t2_plx1.eps")

plt.figure(6)
plt.plot(gaia,dec_plx1*1e-3,linewidth=2)
plt.errorbar(g_sim-2456658.5,dec_plx_sim1*1e-3,yerr=dec_p_simerr*1e-3,fmt='o')
plt.xlabel('JD - 2458484.5')
plt.ylabel(r'$\Delta$Dec (mas)')
plt.xlim(0.0,1826.0)
plt.ylim(min(dec_plx1*1e-3)-1.0,max(dec_plx1*1e-3)+1.0)
plt.title(r'$\Delta$RA versus Time')
plt.savefig("/Users/ram/Dropbox/ut/2015fa/ast381/hw1/dec_v_t2_plx1.eps")
#plt.savefig("/Users/rmartinez/Dropbox/ut/2015fa/ast381/hw1/dec_v_t2_plx1.eps")

plt.figure(7)
plt.plot(ra_pm1*1e-6,dec_pm1*1e-6,linewidth=2)
plt.errorbar(ra_pm_sim1*1e-6,dec_pm_sim1*1e-6,xerr=ra_p_simerr*1e-6,yerr=dec_p_simerr*1e-6,fmt='o')
plt.xlim(max(ra_pm1*1e-6)+0.01,min(ra_pm1*1e-6)-0.01)
plt.ylim(min(dec_pm1*1e-6)-0.01,max(dec_pm1*1e-6)+0.01)
plt.ylabel(r'$\Delta$Dec (as)')
plt.xlabel(r'$\Delta$RA (as)')
plt.title(r'$\Delta$RA versus $\Delta$Dec')
#plt.savefig("/Users/rmartinez/Dropbox/ut/2015fa/ast381/hw1/dec_v_ra3_plx1.eps")
plt.savefig("/Users/ram/Dropbox/ut/2015fa/ast381/hw1/dec_v_ra3_plx1.eps")

plt.figure(8)
plt.errorbar(g_sim-2456658.5,ra_pm_sim1*1e-6,yerr=ra_p_simerr*1e-6,fmt='o')
plt.plot(gaia,ra_pm1*1e-6,linewidth=2)
plt.xlabel('JD-2458484.5')
plt.ylabel(r'$\Delta$RA (as)')
plt.xlim(0.0,1826.0)
plt.ylim(min(ra_pm1*1e-6)-0.01,max(ra_pm1*1e-6)+0.01)
plt.title(r'$\Delta$RA versus Time')
plt.savefig("/Users/ram/Dropbox/ut/2015fa/ast381/hw1/ra_v_t3_plx1.eps")
#plt.savefig("/Users/rmartinez/Dropbox/ut/2015fa/ast381/hw1/ra_v_t3_plx1.eps")

plt.figure(9)
plt.errorbar(g_sim-2456658.5,dec_pm_sim1*1e-6,yerr=dec_p_simerr*1e-6,fmt='o')
plt.plot(gaia,dec_pm1*1e-6,linewidth=2)
plt.xlabel('JD-2458484.5')
plt.ylabel(r'$\Delta$Dec (as)')
plt.xlim(0.0,1826.0)
plt.ylim(min(dec_pm1*1e-6)-0.01,max(dec_pm1*1e-6)+0.01)
plt.title(r'$\Delta$RA versus Time')
#plt.savefig("/Users/rmartinez/Dropbox/ut/2015fa/ast381/hw1/dec_v_t3_plx1.eps")
plt.savefig("/Users/ram/Dropbox/ut/2015fa/ast381/hw1/dec_v_t3_plx1.eps")

#PLX2 Plots
#plt.figure(10)
#plt.errorbar(ra_p_sim2,dec_p_sim2,xerr=ra_p_simerr,yerr=dec_p_simerr,fmt='o')
#plt.plot(ra_p2,dec_p2)
#plt.xlim(max(ra_p2)+0.5,min(ra_p2)-0.5)
#plt.ylim(min(dec_p2)-1.0,max(dec_p2)+1.0)
#plt.ylabel(r'Dec - '+str(dec2000_deg)+' ($\mu$as)')
#plt.xlabel(r'RA - '+str(ra2000_deg)+' ($\mu$as)')
#plt.title('RA versus Dec')
#plt.savefig("/Users/rmartinez/Dropbox/ut/2015fa/ast381/hw1/dec_v_ra1_plx2.eps")
#plt.savefig("/Users/ram/Dropbox/ut/2015fa/ast381/hw1/dec_v_ra1_plx2.eps")

#plt.figure(11)
#plt.errorbar(g_sim-2456658.5,ra_p_sim2,yerr=ra_p_simerr,fmt='o')
#plt.plot(gaia,ra_p2)
#plt.xlabel('JD-2458484.5')
#plt.ylabel(r'RA - '+str(ra2000_deg)+' ($\mu$as)')
#plt.xlim(0.0,1826.0)
#plt.ylim(min(ra_p2)-0.5,max(ra_p2)+0.5)
#plt.savefig("/Users/rmartinez/Dropbox/ut/2015fa/ast381/hw1/ra_v_t1_plx2.eps")
#plt.savefig("/Users/ram/Dropbox/ut/2015fa/ast381/hw1/ra_v_t1_plx2.eps")

#plt.figure(12)
#plt.errorbar(g_sim-2456658.5,dec_p_sim2,yerr=dec_p_simerr,fmt='o')
#plt.plot(gaia,dec_p2)
#plt.xlabel('JD-2458484.5')
#plt.ylabel(r'Dec - '+str(dec2000_deg)+' ($\mu$as)')
#plt.xlim(0.0,1826.0)
#plt.ylim(min(dec_p2)-1.0,max(dec_p2)+1.0)
#plt.savefig("/Users/rmartinez/Dropbox/ut/2015fa/ast381/hw1/dec_v_t1_plx2.eps")
#plt.savefig("/Users/ram/Dropbox/ut/2015fa/ast381/hw1/dec_v_t1_plx2.eps")

#plt.figure(13)
#plt.errorbar(ra_plx_sim2,dec_plx_sim2,xerr=ra_p_simerr,yerr=dec_p_simerr,fmt='o')
#plt.plot(ra_plx2,dec_plx2)
#plt.xlim(max(ra_plx2)+500.0,min(ra_plx2)-500.0)
#plt.ylim(min(dec_plx2)-500.0,max(dec_plx2)+500.0)
#plt.ylabel(r'Dec - '+str(dec2000_deg)+' ($\mu$as)')
#plt.xlabel(r'RA - '+str(ra2000_deg)+' ($\mu$as)')
#plt.title('RA versus Dec')
#plt.savefig("/Users/rmartinez/Dropbox/ut/2015fa/ast381/hw1/dec_v_ra2_plx2.eps")
#plt.savefig("/Users/ram/Dropbox/ut/2015fa/ast381/hw1/dec_v_ra2_plx2.eps")

#plt.figure(14)
#plt.errorbar(g_sim-2456658.5,ra_plx_sim2,yerr=ra_p_simerr,fmt='o')
#plt.plot(gaia,ra_plx2)
#plt.xlabel('JD-2458484.5')
#plt.ylabel(r'RA - '+str(ra2000_deg)+' ($\mu$as)')
#plt.xlim(0.0,1826.0)
#plt.ylim(min(ra_plx2)-500.0,max(ra_plx2)+500.0)
#plt.savefig("/Users/rmartinez/Dropbox/ut/2015fa/ast381/hw1/ra_v_t2_plx2.eps")
#plt.savefig("/Users/ram/Dropbox/ut/2015fa/ast381/hw1/ra_v_t2_plx2.eps")

#plt.figure(15)
#plt.errorbar(g_sim-2456658.5,dec_plx_sim2,yerr=dec_p_simerr,fmt='o')
#plt.plot(gaia,dec_plx2)
#plt.xlabel('JD-2458484.5')
#plt.ylabel(r'Dec - '+str(dec2000_deg)+' ($\mu$as)')
#plt.xlim(0.0,1826.0)
#plt.ylim(min(dec_plx2)-500.0,max(dec_plx2)+500.0)
#plt.savefig("/Users/rmartinez/Dropbox/ut/2015fa/ast381/hw1/dec_v_t2_plx2.eps")
#plt.savefig("/Users/ram/Dropbox/ut/2015fa/ast381/hw1/dec_v_t2_plx2.eps")

plt.figure(16)
plt.errorbar(ra_pm_sim2*1e-6,dec_pm_sim2*1e-6,xerr=ra_p_simerr*1e-6,yerr=dec_p_simerr*1e-6,fmt='o')
plt.plot(ra_pm2*1e-6,dec_pm2*1e-6,linewidth=2)
plt.xlim(max(ra_pm2*1e-6)+0.01,min(ra_pm2*1e-6)-0.01)
plt.ylim(min(dec_pm2*1e-6)-0.01,max(dec_pm2*1e-6)+0.01)
plt.ylabel(r'$\Delta$Dec (as)')
plt.xlabel(r'$\Delta$RA (as)')
plt.title(r'$\Delta$RA versus $\Delta$Dec')
#plt.savefig("/Users/rmartinez/Dropbox/ut/2015fa/ast381/hw1/dec_v_ra3_plx2.eps")
plt.savefig("/Users/ram/Dropbox/ut/2015fa/ast381/hw1/dec_v_ra3_plx2.eps")

#plt.figure(17)
#plt.errorbar(g_sim-2456658.5,ra_pm_sim2,yerr=ra_p_simerr,fmt='o')
#plt.plot(gaia,ra_pm2)
#plt.xlabel('JD-2458484.5')
#plt.ylabel(r'RA - '+str(ra2000_deg)+' ($\mu$as)')
#plt.xlim(0.0,1826.0)
#plt.ylim(min(ra_pm2)-1.0,max(ra_pm2)+1.0)
#plt.savefig("/Users/rmartinez/Dropbox/ut/2015fa/ast381/hw1/ra_v_t3_plx2.eps")
#plt.savefig("/Users/ram/Dropbox/ut/2015fa/ast381/hw1/ra_v_t3_plx2.eps")

#plt.figure(18)
#plt.errorbar(g_sim-2456658.5,dec_pm_sim2,yerr=dec_p_simerr,fmt='o')
#plt.plot(gaia,dec_pm2)
#plt.xlabel('JD-2458484.5')
#plt.ylabel(r'Dec - '+str(dec2000_deg)+' ($\mu$as)')
#plt.xlim(0.0,1826.0)
#plt.ylim(min(dec_pm2)-1.0,max(dec_pm2)+1.0)
#plt.savefig("/Users/rmartinez/Dropbox/ut/2015fa/ast381/hw1/dec_v_t3_plx2.eps")
#plt.savefig("/Users/ram/Dropbox/ut/2015fa/ast381/hw1/dec_v_t3_plx2.eps")