import numpy as np
import ROOT
import matplotlib.pyplot as plt
import scipy.optimize as spyopt
import scipy.special 
import sys
import os.path
import getopt
from numpy import cos, sin, pi

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
#Fermi Function for Sr_90 beta emission spectrum for curve-fitting
#

m_e = 511.
c = 2.998e8
alpha = 1/137
Eul = 0.57722
Z=39
R=0.01395
Q = 545.9


def mom(T_e):
    p=np.sqrt(np.power(W(T_e),2)-1)
    return p

def W(T_e):
    W = (T_e/m_e) + 1
    return W

def Ferm(T_e):
    #print(scipy.special.gamma(complex_fun(gam(Z),y(T_e,mom(T_e)))))#*scipy.special.gamma(complex_fun(gam(Z),-y(T_e,mom(T_e)))))
    Fermi = 2*(1+gam(Z))*np.power(2*mom(T_e)*R,-2*(1-gam(Z)))
    Fermi = Fermi*abs(((scipy.special.gamma(complex_fun(gam(Z),y(T_e))))*(scipy.special.gamma(complex_fun(gam(Z),-y(T_e))))))
    Fermi = Fermi*np.power(scipy.special.gamma(2*gam(Z)+1),-2)
    Fermi = Fermi*np.exp(np.pi*y(T_e))
    return Fermi

def y(T_e):
    y = (alpha*Z*W(T_e))/mom(T_e)
    return y

def complex_fun(g,p):
    if (len(p)>1):
        c = []
        for i in range(0,len(p)):
            c.append(complex(g,i))
        return c
    else:
        c = complex(g,p)
        return c 

def gam(Z_n):
    gam = np.sqrt(1-np.power(alpha*Z_n,2))
    return gam

def q(T_e):
    q = np.power((Q - T_e)/m_e,2)
    return q

def shape_factor(T_e):
    C = 1-0.054*W(T_e)
    return C

def Sr_90_spec(x,counts):
    counts = mom(x)*q(x)*W(x)*shape_factor(W(x))*Ferm(x)*((q(x)+0.85*np.power(mom(x),2)))
    return counts
#
#-------------------------------------------------------------------------------
#
def main(argv):
    
    file_name = '../data/ucna_track_test'

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'f:')
    except getopt.GetoptError as err:
        print(err)
    for r,l in opts:
        if r == '-f':
            file_name = l
    resx = []
    resy = []
    electron_energy = []
    x0p = []
    y0p = []
    pe_spec = []
    unweight_spec = []
    xp = []
    yp = []
    unweight_hits_total = np.zeros(128)
    sipm_hits_total = np.zeros(128)
    index = 0

    xmap,ymap = build_map()
    sipm_locs = calc_sipm_pos()
    theta = np.linspace(0+np.pi/180.,2*np.pi+np.pi/128,128,endpoint=False)
    bottom = 8
    width = (2.*np.pi)/128.
    
    if os.path.exists("data_text_files/{}.txt".format(file_name)) == True:
        read = open("data_text_files/{}.txt".format(file_name),"r")
        for line in read:
            counts = []
            unweight_counts = []
            if 'keV' in line:
                line=line.strip()
                index += 1
                print(index)
                line_seg = line.split(' ')
                electron_energy.append(float(line_seg[0].strip('keV')))
                x0p.append(float(line_seg[1]))
                y0p.append(float(line_seg[2]))
                xp.append(float(line_seg[3]))
                yp.append(float(line_seg[4]))
                resx.append(float(line_seg[5]))
                resy.append(float(line_seg[6]))
                for i in range(0,128):
                    counts.append(float(line_seg[i+7]))
                    unweight_counts.append(float(line_seg[i+135]))
                    sipm_hits_total[i] += float(line_seg[i+7])
                    unweight_hits_total[i] += float(line_seg[i+135])
                pe_spec.append(np.sum(counts))
                unweight_spec.append(np.sum(unweight_counts))

    else:
        #nrun = 0
        #if len(argv) > 1 :
            #nrun = int(argv[1])
    
        fdata = ROOT.TFile("/media/Data_Storage_3/richard_UCNA+/{}.root".format(file_name),"READ")
        #"../data/ucna_track_test.root","READ")
        #"/media/Data_Storage_3/richard_UCNA+/track_test.root","READ"
        #"../data/ucna_test_iso_150_r{}_1125.root".format(nrun),"READ")
        tr = fdata.Get("t")
        N = tr.GetEntries()
        tr.Print()
    
        sipm_pde = open("pde_data","r")
        energy = []
        prob_weight = []
        for line in sipm_pde:
            line = line.strip()
            line_seg = line.split(",")
            energy.append(float(line_seg[0]))
            prob_weight.append(float(line_seg[1]))

        for x in range(0,len(energy)):
            energy[x] = (1.2285)/energy[x]

        for z in range(0,len(prob_weight)):
            prob_weight[z] = prob_weight[z]/100

        #hittree = ROOT.TTree("hitss","SiPM Hits")
        hits = np.empty((128),dtype="float32")
        #hittree.Branch("hits",hits,"hits[128]/F")
   
    
        for evn in tr:
            print(evn.n)
            if index % 100 == 0:
                print("At event : ",index)
        
            sipm_hits  = np.zeros(128)
            unweight_hits = np.zeros(128)
            sipm_quads = np.zeros(16)
            xe = []
            ye = []
            ee = []
            sipm_pos_x = 0
            sipm_pos_y = 0
            
            x0=evn.x[0]/10
            y0=evn.y[0]/10
            
            x0p.append(x0)
            y0p.append(y0)
        

            for i in range(0,tr.n):
                if evn.pdg[i] == 11:
                    xe.append(evn.x[i])
                    ye.append(evn.y[i])
                    ee.append(evn.de[i])
                if evn.vlm[i] >= 100:
                    if evn.pdg[i] == 0 and evn.pro[i]==3031 and evn.de[i]>0:
                        nsipm = evn.vlm[i] - 100
                        ene = evn.de[i]
                        unweight_hits[nsipm] = unweight_hits[nsipm] + 1 
                        for x in range(1,len(energy)-1):
                            if ene>energy[x+1]:
                                if ene<energy[x-1]:
                                    sipm_hits[nsipm] = sipm_hits[nsipm] + prob_weight[x]
                                    quad = int(nsipm/8)
                                    sipm_quads[quad] += prob_weight[x]
            
            hits = sipm_hits
            unwt_hits = unweight_hits
            xe = np.asarray(xe)
            ye = np.asarray(ye)
            
            electron_energy = np.sum(ee)

            xe_avg = np.sum(xe)/len(xe)
            ye_avg = np.sum(ye)/len(ye)
                        
            sipm_pos_x = np.sum(sipm_hits*xmap)/np.sum(sipm_hits)
            sipm_pos_y = np.sum(sipm_hits*ymap)/np.sum(sipm_hits)

            for i in range(0,128):
                sipm_locs[i+256] = sipm_hits[i]
            
            sipm_hits_total = sipm_hits_total + sipm_hits
            unweight_hits_total = unweight_hits_total + unweight_hits
        
            # get the total number of sipm hits
            pe_spec.append(np.sum(hits))
            unweight_spec.append(np.sum(unweight_hits))
        

            res = spyopt.minimize(Weights,x0=(2*sipm_pos_x,2*sipm_pos_y,sum(sipm_hits)),
                              args=(sipm_locs),
                              method='CG')
            
            resx.append(res.x[0])
            resy.append(res.x[1])

            xp.append(res.x[0]-xe_avg/10.)
            yp.append(res.x[1]-ye_avg/10.)
            
            new_file = open("data_text_files/{}.txt".format(file_name),"a")
            new_file.write("{}keV {} {} {} {} {} {} ".format(electron_energy,x0,y0,xp[index],yp[index],resx[index],resy[index]))
            for i in range(0,len(hits)):
                new_file.write("{} ".format(hits[i]))
            for i in range(0,len(hits)):
                new_file.write("{} ".format(unweight_hits[i]))
            new_file.write("\n")
            new_file.close()
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
    nbins = 100
    npe,bpe,pp=ax0.hist(pe_spec,bins=1000,range=(0,5000),histtype='step',color='r',label='Weighted')
    npe_2,bpe_2,pp_2=ax0.hist(unweight_spec,bins=1000,range=(0,5000),histtype='step',color='b',label='Unweighted')

    ebin=[]
    
    for i in range(1,len(bpe)):
        ebin.append((bpe[i]+bpe[i-1])/2.)

    def calc_stddev(data):
        data_avg = np.sum(data)/float(len(data))
        num = []
        for i in range(0,len(data)):
            num.append((data[i]-data_avg)**2)
        sig = np.sqrt(np.sum(num)/(len(data)-1))
        return sig

    def find_max(y,x):
        y_max = y.max()
        for i in range(0,len(y)):
            if y[i]==y_max:
                print(x[i])
                print(y[i])
                return x[i]

    pe_spec_avg = np.sum(pe_spec)/len(pe_spec)
    print(pe_spec_avg)

    unwt_spec_avg = np.sum(unweight_spec)/len(unweight_spec)
    print(unwt_spec_avg)
    
    poptpe,pcovpe = spyopt.curve_fit(peak,ebin,npe,p0=(npe.max(),pe_spec_avg,calc_stddev(pe_spec)))
    poptpe_unw,pcovpe_unw = spyopt.curve_fit(peak,ebin,npe_2,p0=(npe_2.max(),unwt_spec_avg,calc_stddev(unweight_spec)))
    ##first argument was 'peak' function
    ##ax0.plot(bpe,peak(bpe,*poptpe)
    print(poptpe)
    print(poptpe_unw)
    ax0.plot(bpe,peak(bpe,*poptpe),'--',color='tab:red',
            label=r'$\langle N \rangle$ = {:4.3f},$\sigma =$ {:4.3f}'.format(poptpe[1],poptpe[2])) #$\sigma =${:4.3f}'.format(poptpe[1],poptpe[2]))
    ax0.plot(bpe,peak(bpe,*poptpe_unw),'--',color='tab:blue',label=r'$\langle N \rangle$ = {:4.3f},$\sigma =$ {:4.3f}'.format(poptpe_unw[1],poptpe_unw[2])) 
    ax0.set_xlabel("N-Photons")
    ax0.set_ylabel("Counts")
    ax0.legend()

    ax1 = plt.subplot(222)
    ax1.hist2d(resx,resy,nbins,range=((-plimit,plimit),(-plimit,plimit)))
    poly_x = np.zeros(17)
    poly_y = np.zeros(17)
    for i in range(17):
        poly_x[i] = 8.626*cos(2*pi*(i)/16)
        poly_y[i] = 8.626*sin(2*pi*(i)/16)
    #ax1.scatter(poly_x,poly_y)
    ax1.plot(poly_x,poly_y,color='r')
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

    poptx,pcovx = spyopt.curve_fit(peak,bxm,nx)#,sigma=np.sqrt(nx))
    popty,pcovy = spyopt.curve_fit(peak,bym,ny)#,sigma=np.sqrt(ny))

    #ax3.plot(bxm,peak(bxm,*poptx),'--',color='tab:blue',
    #         label=r'$\langle \Delta x \rangle$ = {:4.3f} , $\sigma_x=$ {:4.3f}'.format(poptx[1],poptx[2]))
    #ax3.plot(bym,peak(bym,*popty),'--',color='tab:orange',
    #         label=r'$\langle \Delta y \rangle$ = {:4.3f} , $\sigma_y=$ {:4.3f}'.format(popty[1],popty[2]))
    
    #ax3.plot(bxm,peak(bxm,*poptx),'--')
    #ax3.plot(bym,peak(bym,*popty),'--')

    ax3.errorbar(bxm,nx,yerr=np.sqrt(nx),color='tab:blue',fmt='.')
    ax3.errorbar(bym,ny,yerr=np.sqrt(ny),color='tab:orange',fmt='.')
    
    ax3.legend([r"$\sigma_x$ = {:.4f} cm".format(poptx[2]),r"$\sigma_y$ = {:.4f} cm".format(popty[2])])
    plt.show()
        

if __name__=="__main__":
    main(sys.argv)
