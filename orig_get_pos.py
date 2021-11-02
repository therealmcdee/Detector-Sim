import numpy as np
import ROOT
import matplotlib.pyplot as plt
import scipy.optimize as spyopt
import sys

def Weights(x,locs):

    wghts=[]
    alen  = 360.
    
    for i in range(0,256,2):
        vx0 = locs[i] - x[0]
        vy0 = locs[i+1] - x[1]
        
        if i < 254 :
            vx1 = locs[i+2] - x[0]
            vy1 = locs[i+3] - x[1]
        else:
            vx1 = locs[0] - x[0]
            vy1 = locs[1] - x[1]
            
        mag1 = np.sqrt(vx0*vx0 + vy0*vy0)
        mag2 = np.sqrt(vx1*vx1 + vy1*vy1)
        
        
        v1dotv2 = (vx0*vx1 + vy0*vy1)/(mag1*mag2)
        aveg  = (mag1+mag2)/2.
        v1dotv2 = x[2]*np.arccos(v1dotv2)/(2.*np.pi) * np.exp(-aveg/alen)
        wghts.append(v1dotv2)
        

    wghts = np.asarray(wghts)
    hits = np.zeros(128)
    df  =0
    
    for i in range(0,128):
        df = df + np.power((wghts[i] - locs[256+i]),2)

    df = df/128.

    return df
#------------------------------------------------------------------------------
def calc_sipm_pos():
    pos = np.zeros(128*3)
    
    dx    = 0.429 # SiPM edges
    theta = 11.25 # half opening angle of the wedge
    rad   = 8.794 # center to corner distance

    for i in range(0,16):
        comp_ang = (theta*(1+2*i)/180. + 1/2.)*np.pi
        for j in range(0,8):
            x = rad * np.cos(i*2*theta*np.pi/180.) + dx*j*np.cos(comp_ang)
            y = rad * np.sin(i*2*theta*np.pi/180.) + dx*j*np.sin(comp_ang)
            pos[(i*8+j)*2] = x
            pos[(i*8+j)*2+1] = y
    return pos
#---------------------------------------------------------------------------------
# Simple 1-d 3 parameter Gaussian function for fitting peaks
#
# a = Norm
# b = peak center
# c = peak width
#
def peak(x,a,b,c):
    return a * np.exp(-np.power((x-b),2)/(c*c*2))
#---------------------------------------------------------------------------------
def output_quad(ds):
    print("Quad hits: \n")
#    for i in range(0,len(ds)):
#        print(" q{} , hits {} \n".format(i,ds[i]))   

    return
#---------------------------------------------------------------------------------
#
#  Build the position map from the geometry generation output file.
#
def build_map():
    x = []
    y = []
    
    ft = open("sipm_map.txt","r")

    for line in ft:
        ln = line.split()
        x.append(float(ln[1]))
        y.append(float(ln[2]))

    ft.close()
    x = np.asarray(x)
    y = np.asarray(y)

    return x,y
#-----------------------------------------------------------------------------------
#
def main(argv):

    nrun = 0
    if len(argv) > 1 :
        nrun = int(argv[1])
    
    fdata = ROOT.TFile("../data/geo_test.root","READ") #"../data/ucna_test_iso_150_r{}_1125.root".format(nrun),"READ")
    #("../data/ucna_track_test.root","READ"
    tr = fdata.Get("t")
    N = tr.GetEntries()
    tr.Print()
    
    #hittree = ROOT.TTree("hitss","SiPM Hits")
    hits = np.empty((128),dtype="float32")
    #hittree.Branch("hits",hits,"hits[128]/F")
   
    xmap,ymap = build_map()
    sipm_locs = calc_sipm_pos()
    theta = np.linspace(0+np.pi/180.,2*np.pi+np.pi/128,128,endpoint=False)
    bottom = 8
    width = (2.*np.pi)/128.
    
    pe_spec = []
    xp  = []
    yp  = []
    x0p = []
    y0p = []

    sipm_hits_total = np.zeros(128)
    index = 0
    
    for evn in tr:
        print(evn.n)
        if index % 100 == 0:
            print("At event : ",index)
        
        sipm_hits  = np.zeros(128)
        sipm_quads = np.zeros(16)
        xe = []
        ye = []
        sipm_pos_x = 0
        sipm_pos_y = 0
            
        x0=evn.x[0]/10
        y0=evn.y[0]/10
            
        x0p.append(x0)
        y0p.append(y0)
        
        for i in range(0,tr.n):
            if evn.pdg[i] == 11:
                #print(index,evn.vlm[i],evn.x[i],evn.y[i],evn.z[i],evn.de[i])
                xe.append(evn.x[i])
                ye.append(evn.y[i])
            if evn.vlm[i] >= 100:
                if evn.pdg[i] == 0 and evn.pro[i]==3031 and evn.de[i]>0:
                    nsipm = evn.vlm[i] - 100
                    sipm_hits[nsipm] = sipm_hits[nsipm] + 1
                    quad = int(nsipm/8)
                    sipm_quads[quad] += 1

        hits = sipm_hits
        xe = np.asarray(xe)
        ye = np.asarray(ye)
        
        xe_avg = np.sum(xe)/len(xe)
        ye_avg = np.sum(ye)/len(ye)
                        
        sipm_pos_x = np.sum(sipm_hits*xmap)/np.sum(sipm_hits)
        sipm_pos_y = np.sum(sipm_hits*ymap)/np.sum(sipm_hits)

        for i in range(0,128):
            sipm_locs[i+256] = sipm_hits[i]
        
        sipm_hits_total = sipm_hits_total + sipm_hits
        # get the total number of sipm hits
        pe_spec.append(np.sum(sipm_hits))
        
        res = spyopt.minimize(Weights,x0=(2*sipm_pos_x,2*sipm_pos_y,sum(sipm_hits)),
                              args=(sipm_locs),
                              method='CG')
        
        xp.append(res.x[0]-xe_avg/10.)
        yp.append(res.x[1]-ye_avg/10.)
        index += 1
        
    #------------------------------------------------------------------------------
    #hittree.Write()
    #fdata.Write("",TFile.kOverWrite)
    fdata.Close()

    
    
    print("Average x : ",sum(x0p)/len(x0p))
    print("Average y : ",sum(y0p)/len(y0p))
    xavg = np.sum(xmap*sipm_hits_total)/np.sum(sipm_hits_total)
    print("average x ",xavg)

    fig = plt.figure(figsize=(10,10))
    ax0 = plt.subplot(221)
    plimit = 10
    nbins = 50
    npe,bpe,pp=ax0.hist(pe_spec,histtype='step',bins=1000,range=(0,5000))
    ebin=[]
    
    for i in range(1,len(bpe)):
        ebin.append((bpe[i]+bpe[i-1])/2.)
        
    poptpe,pcovpe = spyopt.curve_fit(peak,ebin,npe)#,p0=(50,500,20))
    print(poptpe)
    ax0.plot(bpe,peak(bpe,*poptpe),'--',color='tab:blue',
             label=r'$\langle N \rangle$ = {:4.3f}, $\sigma =${:4.3f}'.format(poptpe[1],poptpe[2]))

    ax0.set_xlabel("N-Photons")
    ax0.set_ylabel("Counts")
    ax0.legend()

    ax1 = plt.subplot(222)
    ax1.hist2d(xp,yp,nbins,range=((-plimit,plimit),(-plimit,plimit)))
    ax1.set_xlabel(r"$X_{recon}$ [cm]")
    ax1.set_ylabel(r"$Y_{recon}$ [cm]")

    ax2 = plt.subplot(223,projection='polar')
    sipm_hits_total = sipm_hits_total/index
    ax2.bar(theta,sipm_hits_total,width=width,bottom=bottom)
    ax2.set_thetagrids(np.linspace(22.5,360,16))
        
    ax3 = plt.subplot(224)

    nx,bx,pp = ax3.hist(xp,histtype='step',bins=nbins,color='tab:blue')
    ny,by,pp = ax3.hist(yp,histtype='step',bins=nbins,color='tab:orange')

    
    ax3.set_xlabel('Position [cm]')
    ax3.set_ylabel('Counts')
    
    bxm = 0.5*(bx[1:]+bx[:-1])
    bym = 0.5*(by[1:]+by[:-1])

    #poptx,pcovx = spyopt.curve_fit(peak,bxm,nx)#,sigma=np.sqrt(nx))
    #popty,pcovy = spyopt.curve_fit(peak,bym,ny)#,sigma=np.sqrt(ny))

    #ax3.plot(bxm,peak(bxm,*poptx),'--',color='tab:blue',
    #         label=r'$\langle \Delta x \rangle$ = {:4.3f} , $\sigma_x=$ {:4.3f}'.format(poptx[1],poptx[2]))
    #ax3.plot(bym,peak(bym,*popty),'--',color='tab:orange',
    #         label=r'$\langle \Delta y \rangle$ = {:4.3f} , $\sigma_y=$ {:4.3f}'.format(popty[1],popty[2]))

    ax3.errorbar(bxm,nx,yerr=np.sqrt(nx),color='tab:blue',fmt='.')
    ax3.errorbar(bym,ny,yerr=np.sqrt(ny),color='tab:orange',fmt='.')
    
    ax3.legend()
    plt.show()
        

if __name__=="__main__":
    main(sys.argv)
