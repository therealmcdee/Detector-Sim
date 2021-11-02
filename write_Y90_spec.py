import matplotlib.pyplot as plt
import numpy as np
import os 
import sys
import scipy.special

num = 500
T = np.linspace(1,2300,num)
counts = []
m_e = 511.
c = 2.998e8
alpha = 1/137
Eul = 0.57722

#Sr-90
Z=40
R=0.01395
Q = 2278.5


def mom(T_e):
    p=np.sqrt(np.power(W(T_e),2)-1)
    return p

def W(T_e):
    W = (T_e/m_e) + 1
    return W

def y(T_e):
    y = (alpha*Z*W(T_e))/mom(T_e)
    return y

def complex_fun(g,p):
    c = complex(g,p)
    return c

def gam(Z_n):
    gam = np.sqrt(1-np.power(alpha*Z_n,2))
    return gam

def Ferm(T_e):
    #print(scipy.special.gamma(complex_fun(gam(Z),y(T_e,mom(T_e)))))#*scipy.special.gamma(complex_fun(gam(Z),-y(T_e,mom(T_e)))))
    Fermi = 2*(1+gam(Z))*np.power(2*mom(T_e)*R,-2*(1-gam(Z)))
    Fermi = Fermi*abs(((scipy.special.gamma(complex_fun(gam(Z),y(T_e))))*(scipy.special.gamma(complex_fun(gam(Z),-y(T_e))))))
    Fermi = Fermi*np.power(scipy.special.gamma(2*gam(Z)+1),-2)
    Fermi = Fermi*np.exp(np.pi*y(T_e))
    return Fermi

def q(T_e):
    q = np.power((Q - T_e)/m_e,2)
    return q

def shape_factor(T_e):
    C = 1-0.054*W(T_e)
    return C

#print(scipy.special.gamma(complex_fun(gam(Z),y(T[0])))*scipy.special.gamma(complex_fun(gam(Z),-y(T[0]))))

for i in T:
    if i <= Q:
        result = mom(i)*q(i)*W(i)*shape_factor(W(i))*Ferm(i)*((q(i)+0.85*np.power(mom(i),2)))
    else:
        result=0
    counts.append(result)




counts = np.asarray(counts)
counts = counts/np.sum(counts)

plt.plot(T,counts)
plt.show()

counts=counts.tolist()
T = T.tolist()



def list_to_string(inpt):
    mystr = ()
    mystr=' '.join(map(str,inpt))
    return mystr

def add_units(inpt):
    for i in range(len(inpt)):
        inpt[i] = np.round(inpt[i])/1000
    return inpt
def round_inten(inpt):
    for i in range(len(inpt)):
        inpt[i]=np.round(inpt[i],decimals=8)


def write_to_file(oldfile,write_file,intensity,energy,bins):
    old_file = open("/home/mcdonaldre/UCNA+/"+oldfile+"","r")
    lines = old_file.readlines()
    old_file.close()
    newfile = open("/home/mcdonaldre/UCNA+/"+write_file+"","w+")
    for line in lines:
        if ("/gps/source/intensity" in line):
            #line = line.split(" ")
            #line = line[1].strip()
            line = line.replace(line,"\n")
            newfile.write(line)
        elif ("/gps/energy" in line):
            #line = line.split(" ")
            #line = line[1].strip()
            #add_units(energy)
            line = line.replace(line,'/gps/ene/type User\n/gps/hist/type energy\n')
            newfile.write(line)
            add_units(energy)
            round_inten(intensity)
            for i in range(len(energy)):
                newfile.write("/gps/hist/point {} {}\n".format(energy[i],intensity[i]))
        else:
            newfile.write(line)
    newfile.close()

write_to_file("rm_test.mac","Y90.mac",counts,T,num)
