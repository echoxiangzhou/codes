#!/usr/bin/env python
#
"""Return a png file with a plot of the 2D wavenumber spectrum of a variable. 
"""

"""todo = 
 - ajout indication en km (cf these nathacha)
"""
## modules
import os, user, sys
import numpy as npy
import numpy.fft as fft
import matplotlib
import matplotlib.pyplot as plt
try:
   import seaborn as sns
   sns_available = True
except:
   sns_available = False

earthrad = 6371229            # mean earth radius (m)
deg2rad = npy.pi / 180.

## core funtions


def read(filein,latmin=None,latmax=None,lonmin=None,lonmax=None,level=None,varname=None,time=0,netcdf=None,**kwargs):
   """Return navlon,navlat,data.
   """
   if netcdf=='3':
      from scipy.io import netcdf 
      ncopen = netcdf.netcdf_file
   elif netcdf=='4':
      from netCDF4 import Dataset as ncopen
   for key in kwargs:
       exec(key + " = kwargs['" + key + "']")
   ncfile = ncopen(filein,'r')
   # get indices
   _navlon = ncfile.variables['nav_lon'][:,:]
   _navlat = ncfile.variables['nav_lat'][:,:]
   if latmin is None: latmin = _navlat.min()
   if latmax is None: latmax = _navlat.max() 
   if lonmin is None: lonmin = _navlon.min()
   if lonmax is None: lonmax = _navlon.max()
   domain = (lonmin<_navlon) * (_navlon<lonmax) * (latmin<_navlat) * (_navlat<latmax)
   where = npy.where(domain)
   vlats = _navlat[where]
   vlons = _navlon[where]
   jmin = where[0][vlats.argmin()]
   jmax = where[0][vlats.argmax()]
   imin = where[1][vlons.argmin()]
   imax = where[1][vlons.argmax()]
   #load arrays
   navlon = _navlon[jmin:jmax+1,imin:imax+1]
   navlat = _navlat[jmin:jmax+1,imin:imax+1]
   if level is None:
      data = ncfile.variables[varname][time,jmin:jmax+1,imin:imax+1]
   else:   
      data = ncfile.variables[varname][time,level,jmin:jmax+1,imin:imax+1]
   return navlon,navlat,data


def e1e2(navlon,navlat):
    """Compute scale factors from navlon,navlat.
    """
    lam = navlon
    phi = navlat
    djlam,dilam = npy.gradient(lam)
    djphi,diphi = npy.gradient(phi)
    e1 = earthrad * deg2rad * npy.sqrt( (dilam * npy.cos(deg2rad * phi))**2. + diphi**2.)
    e2 = earthrad * deg2rad * npy.sqrt( (djlam * npy.cos(deg2rad*phi))**2. + djphi**2.)
    return e1,e2

def test_plot(X,Y,Z):
    plt.pcolormesh(X,Y,Z)
    plt.show()

def interpolate(data,navlon,navlat,interp=None):
    """Perform a spatial interpolation if required; return x_reg,y_reg,data_reg.
    """
    e1,e2 = e1e2(navlon,navlat) # ideally we would like e1u and not e1t...
    x1d_in = e1[0,:].cumsum() - e1[0,0]
    y1d_in = e2[:,0].cumsum() - e2[0,0]
    x2d_in,y2d_in = npy.meshgrid(x1d_in,y1d_in)
    # print x1d_in
    if interp is None or interp=='0':
       return x2d_in, y2d_in, data.copy()
    elif interp=='basemap': # only for rectangular grid...
       from mpl_toolkits import basemap
       x1d_reg=npy.linspace(x1d_in[0],x1d_in[-1],len(x1d_in))
       y1d_reg=npy.linspace(y1d_in[0],y1d_in[-1],len(y1d_in))
       x2d_reg,y2d_reg = npy.meshgrid(x1d_reg,y1d_reg)
       data_reg=basemap.interp(data,x1d_in,y1d_in,x2d_reg,y2d_reg,checkbounds=False,order=1)
       return x2d_reg,y2d_reg,data_reg
    elif interp=='scipy': # only for rectangular grid...
       import scipy.interpolate
       x1d_reg=npy.linspace(x1d_in[0],x1d_in[-1],len(x1d_in))
       y1d_reg=npy.linspace(y1d_in[0],y1d_in[-1],len(y1d_in))
       x2d_reg,y2d_reg = npy.meshgrid(x1d_reg,y1d_reg)
       interp = scipy.interpolate.interp2d(x1d_in, y1d_in,data, kind='linear')
       a1d = interp(x2d_reg[0,:],y2d_reg[:,0])
       data_reg = npy.reshape(a1d,y2d_reg.shape)
       #test_plot(x2d_in,y2d_in,data)
       #test_plot(x2d_reg,y2d_reg,data_reg)
       return x2d_reg,y2d_reg,data_reg

def get_spectrum_1d(data_reg,x_reg,y_reg):
    """Compute the 1d power spectrum.
    """
    # remove the mean and squarize
    data_reg-=data_reg.mean()
    jpj,jpi = data_reg.shape
    msize = min(jpj,jpi)
    data_reg = data_reg[:msize-1,:msize-1]
    x_reg = x_reg[:msize-1,:msize-1]
    y_reg = y_reg[:msize-1,:msize-1]
    # wavenumber vector
    x1dreg,y1dreg = x_reg[0,:],y_reg[:,0]
    Ni,Nj = msize-1,msize-1
    dx=npy.int(npy.ceil(x1dreg[1]-x1dreg[0]))
    k_max  = npy.pi / dx
    kx = fft.fftshift(fft.fftfreq(Ni, d=1./(2.*k_max)))
    ky = fft.fftshift(fft.fftfreq(Nj, d=1./(2.*k_max)))
    kkx, kky = npy.meshgrid( ky, kx )
    Kh = npy.sqrt(kkx**2 + kky**2)
    Nmin  = min(Ni,Nj)
    leng  = Nmin/2+1
    kstep = npy.zeros(leng)
    kstep[0] =  k_max / Nmin
    for ind in range(1, leng):
        kstep[ind] = kstep[ind-1] + 2*k_max/Nmin
    norm_factor = 1./( (Nj*Ni)**2 )
    # tukey windowing = tapered cosine window
    cff_tukey = 0.25
    yw=npy.linspace(0, 1, Nj)
    wdw_j = npy.ones(yw.shape)
    xw=npy.linspace(0, 1, Ni)
    wdw_i= npy.ones(xw.shape)
    first_conditioni = xw<cff_tukey/2
    first_conditionj = yw<cff_tukey/2
    wdw_i[first_conditioni] = 0.5 * (1 + npy.cos(2*npy.pi/cff_tukey * (xw[first_conditioni] - cff_tukey/2) ))
    wdw_j[first_conditionj] = 0.5 * (1 + npy.cos(2*npy.pi/cff_tukey * (yw[first_conditionj] - cff_tukey/2) ))
    third_conditioni = xw>=(1 - cff_tukey/2)
    third_conditionj = yw>=(1 - cff_tukey/2)
    wdw_i[third_conditioni] = 0.5 * (1 + npy.cos(2*npy.pi/cff_tukey * (xw[third_conditioni] - 1 + cff_tukey/2)))
    wdw_j[third_conditionj] = 0.5 * (1 + npy.cos(2*npy.pi/cff_tukey * (yw[third_conditionj] - 1 + cff_tukey/2)))
    wdw_ii, wdw_jj = npy.meshgrid(wdw_j, wdw_i, sparse=True)
    wdw = wdw_ii * wdw_jj
    data_reg*=wdw
    #2D spectrum
    cff  = norm_factor
    tempconj=fft.fft2(data_reg).conj()
    tempamp=cff * npy.real(tempconj*fft.fft2(data_reg))
    spec_2d=fft.fftshift(tempamp)
    #1D spectrum
    leng    = len(kstep)
    spec_1d = npy.zeros(leng)
    krange     = Kh <= kstep[0]
    spec_1d[0] = spec_2d[krange].sum()
    for ind in range(1, leng):
        krange = (kstep[ind-1] < Kh) & (Kh <= kstep[ind])
        spec_1d[ind] = spec_2d[krange].sum()
    spec_1d[0] /= kstep[0]
    for ind in range(1, leng):
        spec_1d[ind] /= kstep[ind]-kstep[ind-1]
    return spec_1d, kstep

def default_title(**kwargs):
    """Return a default title if none is provided. 
    """
    lats = 'lats = ' +  str(kwargs['latmin']) + '-' + str(kwargs['latmax'])
    lons = 'lons = ' +  str(kwargs['lonmin']) + '-' + str(kwargs['lonmax'])
    title = kwargs['varname'] + ' at level ' + str(kwargs['level']) + ', ' + lats + ', ' + lons
    return title

def get_ktuples(klines):
    """Return a list of tuple of (kval,kstr) where kval : float, kstr:string.
    """
    if klines is None or len(klines)==0:
       return []
    else:
       splitted = klines.split(',')
       out = []
       for kstr in splitted:
	   if '/' not in kstr:
              kval = float(kstr)
	   else:
	      nom,denom = kstr.split('/')
              kval = float(nom)/float(denom) 
           out.append((kval,kstr))
       return out

def plot_spectrum(pspec,kstep,**kwargs):
    """Create a png file with the plot.
    Note : kstep is given in rad/m but we plot cycle/km.
    """
    for key in kwargs:
       exec(key + " = kwargs['" + key + "']")
    print "legend",legend       
    if title is None:
       title = default_title(**kwargs)
    rad2cyc = 1.E3 / npy.pi / 2.  # CHECK ME : 2 pi or pi ?  
    kstep*= rad2cyc # kstep is given in rad/m but we plot cycle/km
    fig  = plt.figure()
    ax = plt.subplot(1,1,1)
    # plot power density spectrum
    y_min = 10 ** npy.floor(npy.log10(pspec.min())-1)
    y_max = 10 ** npy.ceil( npy.log10(pspec.max())+1)
    plt.plot(kstep[1:], pspec[1:],'go-', lw=3, label=varname)
    # plotting lines for estimating slopes
    ktuples  = get_ktuples(klines)
    for ktuple in ktuples:
        kval,kstr = ktuple
	lk = len(kstep)
	plt.plot(kstep[lk/20:2*lk/3], 0.1*y_max*(2*kstep[lk/20:2*lk/3]/kstep[lk/20])**(kval), 'k', lw=1.5, label=r'$k^{'+kstr+'}$')
    if kfit:
        kstepmin = km2kstep(klmin)
	kstepmax = km2kstep(klmax)
        kstep_r = kstep[(kstep<kstepmin)*(kstep>kstepmax)]
        pspec_r = pspec[(kstep<kstepmin)*(kstep>kstepmax)]
	kval = estimate_slope(pspec_r,kstep_r)
	kstr = "{:1.1f}".format(kval)
        mpspc = pspec_r.max()
        mkstp = kstep_r[pspec_r == mpspc][0]
	toplt = 10 * mpspc *(kstep_r/kstep_r[0])**(kval)
        plt.plot(kstep_r,  toplt, 'k', lw=1.5, label=r'$k^{'+kstr+'}$')
	logkstp = npy.log(kstep_r)
	logpsp = npy.log(pspec_r)
	xpos = npy.exp((logkstp[0] + logkstp[-1])/2.)
	ypos = npy.exp( (logpsp[0] + logpsp[-1])/2.)*12
	ax.text(xpos,ypos,kstr) # http://matplotlib.org/users/text_props.html
        print "estimated slope : ", kstr
    # formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Wavenumber [cycle/km]',fontsize=16)
    ax.set_xbound(1e-5*rad2cyc, 1e-2*rad2cyc)
    ax.set_ybound(y_min, y_max)
    plt.title(title)
    if legend:
       ax.legend(fontsize=10)
    ax.grid('on')
    plt.show()
    if plotname is not None:
       fig.savefig(plotname)

def kstep2km(k):
    """Convert wavenumbers (cycle/km) to lenghscales (km)"""
    return 1./k

def km2kstep(l):
    """Convert lenghscales (km) to wavenumbers (cycle/km)"""
    return 1./l

def estimate_slope(pspec,kstep):
    """Linear fit estimation of the slope. assumes pspec and kstep have been restricted to the correct interval."""
    # http://stackoverflow.com/questions/21353576/performing-linear-regression-on-a-log-log-base-10-plot-matlab
    pfit = npy.polyfit(npy.log(kstep), npy.log(pspec), 1)
    #p = npy.poly1d(pfit)
    power = pfit[0]
    #coef = pfit[1]
    return power 

## parser and main
def script_parser():
    """Customized parser.
    """
    from optparse import OptionParser
    usage = "usage: %prog [options] file_gridT.nc"
    parser = OptionParser(usage=usage)
    parser.add_option("-v", "--varname", dest="varname",\
                    help="name of the variable to process", default='votemper')
    parser.add_option("--latmin", dest="latmin",type="float",\
                      help="southernmost latitude to process", default=None)
    parser.add_option("--latmax", dest="latmax",type="float",\
                      help="northernmost latitude to process", default=None)
    parser.add_option("--lonmin", dest="lonmin",type="float",\
                      help="westernmost longitude to process", default=None)
    parser.add_option("--lonmax", dest="lonmax",type="float",\
                      help="easternmost longitude to process", default=None)
    parser.add_option("--klmin", dest="klmin",type="float",\
		      help="smallest lengthscale for plotting klines and estimating slopes (km)", default=10.)
    parser.add_option("--klmax", dest="klmax",type="float",\
		      help="largest lengthscale for plotting klines and estimating slopes (km)", default=100.)
    parser.add_option("-l","--level", dest="level",type="int",\
                      help="level of the variable to process, default value for 2D fields", default=None)    
    parser.add_option("-p", "--plotname", dest="plotname",\
                    help="name of the ouput .png file, with -p0 return pspec and kstep", \
		    default='./spectre.png')
    parser.add_option("-s", action="store_true", dest="showmap", default=False,\
		    help="first plot a 2D map of the field to process.")
    parser.add_option("-t", "--title", dest="title",\
                    help="title of the plot", default=None)
    parser.add_option("-k", "--klines", dest="klines",\
                    help="Draw lines k^n with values k1,...,kn", default='')
    parser.add_option("-f","--kfit", action="store_true", dest="kfit", default=False,\
                    help="add a kline from the linear fit estimation of the slope of the PSD.")
    parser.add_option("-i", "--interp", dest="interp",\
		    help="type of interpolation : basemap, scipy; -i0 uses no interpolation", default='basemap')
    parser.add_option("-g", action="store_true", dest="legend", default=False,\
                    help="whether to show the legend.")
    parser.add_option("-n", "--netcdf", dest="netcdf",\
		    help="Netcdf I/O module : 3 for scipy-io, 4 for netCDF4  ", default='4')
    return parser


def main():
    parser = script_parser()
    (options, args) = parser.parse_args()
    if len(args)!=1: # print the help message if number of args is not one.
        parser.print_help()
        sys.exit()
    optdic = vars(options)
    filein = args[0]
    navlon,navlat,data = read(filein,**optdic)
    if optdic['showmap'] is True:
       plt.pcolormesh(navlon,navlat,data)
       plt.colorbar()
       plt.show()
    x_reg,y_reg,data_reg = interpolate(data,navlon,navlat,interp=optdic['interp'])
    pspec,kstep = get_spectrum_1d(data_reg,x_reg,y_reg)
    if optdic['plotname']=='0':
       return pspec,kstep
    else:   
       plot_spectrum(pspec,kstep,**optdic)
       return


if __name__ == '__main__':
    sys.exit(main() or 0)
