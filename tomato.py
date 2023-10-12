import cv2 
import numpy as np 
import matplotlib.pyplot as plt

image = cv2.imread('tomato.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

edged = cv2.Canny(gray, 30, 200) 
cv2.waitKey(0) 
contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
cv2.imshow('Canny Edges After Contouring', edged)
cv2.imwrite('tomato1.jpg', edged)  
cv2.waitKey(0) 

print("Number of Contours: " + str(len(contours))) 
  
cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 

contours2, hiérarchie2= cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 


cv2.imshow('Contours', image)
cv2.imwrite('Contours.jpg', image) 
cv2.waitKey(0) 





image = cv2.imread('tomato1.jpg', cv2.IMREAD_GRAYSCALE)
image=np.invert(image)

ret, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

scatter = np.where(image==0)
x,y = scatter[1], -scatter[0]

for i in range(len(x)):
	print("x=",x[i],"y=",y[i])

	
plt.scatter(x,y,color='black',marker='.',s=0.6)  

z=len(x)
h=z//5
print("le len est ::",z)
print("le h est ::",h)
h_x=[]
h_y=[]
for i in range (h):
    h_x.append(x[i])
    h_y.append(y[i])
h_x1=[]
h_y1=[]

for i in range (h,2*h):
    h_x1.append(x[i])
    h_y1.append(y[i])

h_x2=[]
h_y2=[]
for i in range (2*h,3*h):
    h_x2.append(x[i])
    h_y2.append(y[i])
    
h_x3=[]
h_y3=[]
for i in range (3*h,4*h):
    h_x3.append(x[i])
    h_y3.append(y[i]) 
#300 _ 500
#Y 67_220   
  
h_x4=[]
h_y4=[]
for i in range (4*h,5*h):
    h_x4.append(x[i])
    h_y4.append(y[i])


 
plt.scatter(h_x,h_y ,color='green',marker='+',s=0.6)  
plt.legend()
plt.scatter(h_x1,h_y1 ,color='red',marker='+',s=0.6)
#plt.plot(x,[f(i)for i in x ],label='f(x)',c='red')
#plt.legend()
#plt.plot(h_x,h_y ,color='green',label='1')

#plt.plot(h_x1,h_y1 ,color='red',label='2')
plt.legend()
plt.scatter(h_x2,h_y2 ,c='yellow',marker='+',s=0.6)  
plt.legend() 
plt.scatter(h_x3,h_y3 ,c='pink',marker='+',s=0.6)  
plt.legend() 
plt.scatter(h_x4,h_y4 ,c='black',marker='+',s=0.6)  
#plt.legend() 
#plt.fill_between(x=0,x=2z,alpha=0.4)
#plt.fill_between((1,z),[f(1),f(z)],color='b',alpha=0.4)
plt.show()  
plt.show()

plt.show()



#############################################################
""""moindre caree"""

"""les x et y de newton et moindre caree::"""
h_xin0=[]
h_yin0=[]
for i in range (0,4):
    h_xin0.append(h_x[i+150])
    h_yin0.append(h_y[i+150]) 
fil=[]
p=np.polyfit(h_xin0,h_yin0,3)
for i in h_xin0 :
    s=0
    s=p[0]*i**3+p[1]*i**2+p[2]*i +p[3]
    fil.append(s)

h_xin1=[]
h_yin1=[]
for i in range (0,4):
    h_xin1.append(h_x1[i+150])
    h_yin1.append(h_y1[i+150])
    

fil1=[]
p1=np.polyfit(h_xin1,h_yin1,3)
for i in h_xin1 :
    s=0
    s=p[0]*i**3+p[1]*i**2+p[2]*i +p[3]
    fil1.append(s)
 
"""la 2 font  le len est :: 3087"""
h_xin2=[]
h_yin2=[]
for i in range (0,4):
    h_xin2.append(h_x2[i+150])
    h_yin2.append(h_y2[i+150])
fil2=[]
p1=np.polyfit(h_xin2,h_yin2,3)
for i in h_xin2 :
    s=0
    s=p[0]*i**3+p[1]*i**2+p[2]*i +p[3]
    fil2.append(s)
    
h_xin3=[]
h_yin3=[]
for i in range (0,4):
    h_xin3.append(h_x3[i+150])
    h_yin3.append(h_y3[i+150])
fil3=[]
p1=np.polyfit(h_xin3,h_yin3,3)
for i in h_xin2 :
    s=0
    s=p[0]*i**3+p[1]*i**2+p[2]*i +p[3]
    fil3.append(s)
 
 
h_xin4=[]
h_yin4=[]
for i in range (0,4):
    h_xin4.append(h_x4[i+150])
    h_yin4.append(h_y4[i+150])
fil4=[]
p1=np.polyfit(h_xin4,h_yin4,3)
for i in h_xin4 :
    s=0
    s=p[0]*i**3+p[1]*i**2+p[2]*i +p[3]
    fil4.append(s)

print("le coefficient de la 1° partie est::",fil)
print("le polynome de la 2° partie est ::",fil1)
print("le polynome de la 3° partie est::",fil2)
print("le polynome de la 4° partie est::",fil3)
print("le polynome de la 5° partie est::",fil4) 
#plt.scatter(h_x2,h_y2,label='f(x)',c='y',marker='*')



 



#affichage



plt.subplot(231)
plt.plot(h_xin0,fil,marker='+' ,color='g',label='1')
plt.subplot(232)
plt.plot(h_xin1,fil1,marker='+' ,color='r',label='2')
plt.subplot(233)
plt.plot(h_xin2,fil2,marker='+' ,color='y',label='3')
plt.subplot(234)
plt.plot(h_xin3,fil3,marker='+' ,color='b',label='4')
plt.subplot(235)
plt.plot(h_xin4,fil4,marker='+' ,color='ORANGE',label='5')
plt.show()
"""
"""
 


 
plt.show()
################################################"
#newton
import numpy as np
import matplotlib.pyplot as plt

#plt.style.use('seaborn-poster')


 
def divided_diff(x, y):
    '''
    function to calculate the divided
    differences table
    '''
    n = len(y)
    coef = np.zeros([n, n])
    # the first column is y
    coef[:,0] = y
    
    for j in range(1,n):
        for i in range(n-j):
            coef[i][j] = \
           (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j]-x[i])
            
    return coef

def newton_poly(coef, x_data, x):
    '''
    evaluate the newton polynomial 
    at x
    '''
    n = len(x_data) - 1 
    p = coef[n]
    for k in range(1,n+1):
        p = coef[n-k] + (x -x_data[n-k])*p
    return p

 

# get the divided difference coef
"""1"""
a_s0 = divided_diff(h_xin0, h_yin0)[1,:]

# evaluate on new data points
minim0=min(h_xin0)
maxim0=max(h_xin0)
 
x_new0 = np.arange(minim0,maxim0,0.1) 
y_new0 = newton_poly(a_s0, h_xin0, x_new0)

"""2"""
a_s1 = divided_diff(h_xin1, h_yin1)[1,:]

# evaluate on new data points
minim1=min(h_xin1)
maxim1=max(h_xin1)
 
x_new1 = np.arange(minim1,maxim1,0.1) 
y_new1 = newton_poly(a_s0, h_xin1, x_new1)

"""3"""
a_s2 = divided_diff(h_xin2, h_yin2)[1,:]

# evaluate on new data points
minim2=min(h_xin2)
maxim2=max(h_xin2)
 
x_new2 = np.arange(minim2,maxim2,0.1) 
y_new2 = newton_poly(a_s2, h_xin2, x_new2)

"""4"""
a_s3 = divided_diff(h_xin3, h_yin3)[1,:]

# evaluate on new data points
minim3=min(h_xin3)
maxim3=max(h_xin3)
 
x_new3 = np.arange(minim3,maxim3,0.1) 
y_new3 = newton_poly(a_s3, h_xin3, x_new3)

"""5"""
a_s4 = divided_diff(h_xin4, h_yin4)[1,:]

# evaluate on new data points
minim4=min(h_xin4)
maxim4=max(h_xin4)
 
x_new4 = np.arange(minim4,maxim4,0.1) 
y_new4 = newton_poly(a_s4, h_xin4, x_new4)

"""
print("les polynome de degré au minimum égal à 3 modèle de Newton::")
print("le polynome de la 1° partie est::",y_new0)
print("le polynome de la 2° partie est ::",y_new1)
print("le polynome de la 3° partie est::",y_new2)
print("le polynome de la 4° partie est::",y_new3)
print("le polynome de la 5° partie est::",y_new4) 
#plt.figure(figsize = (12, 8))
"""
#plt.plot(x_new, y_new)
#plt.scatter(h_xin,h_yin ,c='black',marker='+',s=50)
#plt.legend() 
#plt.show()


plt.subplot(231)
plt.plot(x_new0,y_new0,marker='+' ,color='g',label='1')
plt.subplot(232)
plt.plot( x_new1,y_new1,marker='+' ,color='r',label='2')
plt.subplot(233)
plt.plot( x_new2,y_new2,marker='+' ,color='y',label='3')
plt.subplot(234)
plt.plot( x_new3,y_new3,marker='+' ,color='b',label='4')
plt.subplot(235)
plt.plot(x_new4,y_new4,marker='+' ,color='ORANGE',label='5')
plt.show()



#calculer la surface 

#parti 1

a = h_x[150]
b= h_x[153]
n =3
f =s # mettre le polynome s dans la foncton f 
class Simpson ( object ) :
    def __init__ (self , a , b , n , f ) :
        self.a = a
        self.b = b
        self.x = np.linspace( a , b , n+1 )
        self.f = f
        self.n = n   
    def integrate ( self , f ) :
        x = self.x
        y=f(x)
        h=float(x[1]-x[0])
        n=len(x)-1
        if n%2==1:
            n-=1
        s = y[0] + y[n] + 4.0 * sum(y[1:-1:2]) + 2.0 * sum(y[2:-2:2])
        return h*s/3.0
s1 =h*s/3.0
print("la surface du parti 1 est ",-s1)        

# parti 2
a = h_x1[150]
b= h_x1[153]
n =3
f =s # mettre le polynome s dans la foncton f 
class Simpson ( object ) :
    def __init__ (self , a , b , n , f ) :
        self.a = a
        self.b = b
        self.x = np.linspace( a , b , n+1 )
        self.f = f
        self.n = n   
    def integrate ( self , f ) :
        x = self.x
        y=f(x)
        h=float(x[1]-x[0])
        n=len(x)-1
        if n%2==1:
            n-=1
        s = y[0] + y[n] + 4.0 * sum(y[1:-1:2]) + 2.0 * sum(y[2:-2:2])
        return h*s/3.0
s2 =h*s/3.0
print("la surface du parti 2 est ",-s2)


#partie 3
a = h_x2[150]
b= h_x2[153]
n =3
f =s # mettre le polynome s dans la foncton f 
class Simpson ( object ) :
    def __init__ (self , a , b , n , f ) :
        self.a = a
        self.b = b
        self.x = np.linspace( a , b , n+1 )
        self.f = f
        self.n = n   
    def integrate ( self , f ) :
        x = self.x
        y=f(x)
        h=float(x[1]-x[0])
        n=len(x)-1
        if n%2==1:
            n-=1
        s = y[0] + y[n] + 4.0 * sum(y[1:-1:2]) + 2.0 * sum(y[2:-2:2])
        return h*s/3.0
s3 =h*s/3.0
print("la surface du parti 3 est ",-s3)     

#partie 4


a = h_x3[150]
b= h_x3[153]
n =3
f =s # mettre le polynome s dans la foncton f 
class Simpson ( object ) :
    def __init__ (self , a , b , n , f ) :
        self.a = a
        self.b = b
        self.x = np.linspace( a , b , n+1 )
        self.f = f
        self.n = n   
    def integrate ( self , f ) :
        x = self.x
        y=f(x)
        h=float(x[1]-x[0])
        n=len(x)-1
        if n%2==1:
            n-=1
        s = y[0] + y[n] + 4.0 * sum(y[1:-1:2]) + 2.0 * sum(y[2:-2:2])
        return h*s/3.0
s4 =h*s/3.0
print("la surface du parti 4 est ",-s4)  


#partie 5


a = h_x4[150]
b= h_x4[153]
n =3
f =s # mettre le polynome s dans la foncton f 
class Simpson ( object ) :
    def __init__ (self , a , b , n , f ) :
        self.a = a
        self.b = b
        self.x = np.linspace( a , b , n+1 )
        self.f = f
        self.n = n   
    def integrate ( self , f ) :
        x = self.x
        y=f(x)
        h=float(x[1]-x[0])
        n=len(x)-1
        if n%2==1:
            n-=1
        s = y[0] + y[n] + 4.0 * sum(y[1:-1:2]) + 2.0 * sum(y[2:-2:2])
        return h*s/3.0
s5 =h*s/3.0
print("la surface du parti 1 est ",-s5)


print("la surface de tout l'objet est", -(s1+s2+s3+s4+s5))























cv2.destroyAllWindows() 