import matplotlib.pyplot as plt
import numpy as np
import scipy.special

T = np.linspace(1,600,num=100)
counts = []
m_e = 511.
c = 2.998e8
alpha = 1/137
Eul = 0.57722

#Sr-90
Z=39
R=0.01395
Q = 545.9


def mom(T_e):
    p=np.sqrt(np.power(W(T_e),2)-1)
    return p

def W(T_e):
    W = (T_e/m_e) + 1
    return W

def Fermi_appx(Z_n,a_R,T_e):
    Ferm = ()
    #if T_e == 0:
        #a0 = 1
        #a1 = 0
        #a2 = (11/4)-Eul
        #a3 = 0
        #Ferm = a0 + a2*np.power(alpha*Z_n,2)
    #else:
    a0 = 1
    a1 = (np.pi*W(T_e))/mom(T_e)
    a2 = (11/4)-Eul-np.log(2.*mom(T_e)*a_R)+(1/3)*np.power(np.pi,2)*np.power(W(T_e)/(mom(T_e)),2)
    a3 = np.pi*(W(T_e)/(mom(T_e)))*((11/4)-Eul-np.log(2*mom(T_e)*a_R))
    Ferm = a0+(a1*alpha*Z_n)+(a2*np.power(alpha*Z_n,2))+(a3*np.power(alpha*Z_n,3))
    return Ferm

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


avg_x = np.sum(T*counts)/np.sum(counts)

print("Mean T = {}".format(avg_x))

def find_max_T(y_val,x_val):
    y_max = y_val.max()
    y_val = y_val.tolist()
    x_val = x_val.tolist()
    a=len(y_val)
    for b in range(a):
        if y_val[b]==y_max:
            return x_val[b]


print("Most prevalent T = {}".format(find_max_T(counts,T)))
T = np.asarray(T)
#print("Mean T = {}".format(np.mean(counts)))

#print(counts)
#print(T)

print(len(T))

#data = []
#outp = open("Sr_90_spec_data","w+")
#for i in range(0,len(T)):
    #data = str("{} {}\n".format(T[i],counts[i]))
    #outp.write(data)

#print(data)
#outp.write(data)

plt.plot(T,counts)
plt.xlabel('T electron (keV)')
plt.ylabel('Intensity')
plt.title('Sr-90 Beta emission spectrum')
plt.show()
