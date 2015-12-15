import os
import math
import Image

def get_image(data):
    n,m=len(data),len(data[0])
    image=Image.new('L',(n,m))
    for i in xrange(n):
        for j in xrange(m):
            image.putpixel((i,j),data[i][j])
    return image

def roberts(data):
    n,m=len(data),len(data[0])
    ret=[[0]*m for x in xrange(n)]
    for i in xrange(1,n):
        for j in xrange(1,m):
            gx=data[i][j]-data[i-1][j-1]
            gy=data[i][j-1]-data[i-1][j]
            ret[i][j]=abs(gx)+abs(gy)
    return ret

def prewitt(data):
    n,m=len(data),len(data[0])
    ret=[[0]*m for x in xrange(n)]
    for i in xrange(1,n-1):
        for j in xrange(1,m-1):
            gx=data[i+1][j-1]+data[i+1][j+1]+data[i+1][j]-data[i-1][j-1]-data[i-1][j+1]-data[i-1][j]
            gy=data[i-1][j+1]+data[i+1][j+1]+data[i][j+1]-data[i-1][j-1]-data[i+1][j-1]-data[i][j-1]
            ret[i][j]=abs(gx)+abs(gy)
    return ret

def sobel(data):
    n,m=len(data),len(data[0])
    ret=[[0]*m for x in xrange(n)]
    for i in xrange(1,n-1):
        for j in xrange(1,m-1):
            gx=data[i+1][j-1]+data[i+1][j+1]+2*data[i+1][j]-data[i-1][j-1]-data[i-1][j+1]-2*data[i-1][j]
            gy=data[i-1][j+1]+data[i+1][j+1]+2*data[i][j+1]-data[i-1][j-1]-data[i+1][j-1]-2*data[i][j-1]
            ret[i][j]=abs(gx)+abs(gy)
    return ret

def zero_crossing(data):
    n,m=len(data),len(data[0])
    ret=[[0]*m for x in xrange(n)]
    dx=[-1,-1,-1,0]
    dy=[-1,0,1,-1]
    for x in xrange(1,n-1):
        for y in xrange(1,m-1):
            for k in xrange(4):
                x0,x1=x-dx[k],x+dx[k]
                y0,y1=y-dy[k],y+dy[k]
                if data[x0][y0]*data[x1][y1]<0:
                    ret[x][y]=data[x][y]
    return ret

def marr_hildreth(data):
    n,m=len(data),len(data[0])
    tmp=[[0]*m for x in xrange(n)]
    for i in xrange(2,n-2):
        for j in xrange(2,m-2):
            s1=data[i-1][j]+data[i+1][j]+data[i][j-1]+data[i][j+1]
            s2=data[i-2][j]+data[i+2][j]+data[i][j-2]+data[i][j+2]+data[i-1][j-1]+data[i-1][j+1]+data[i+1][j-1]+data[i+1][j+1]
            tmp[i][j]=16*data[i][j]-2*s1-s2
    return zero_crossing(tmp)

def between(a,lo,hi):
    return a>=lo and a<hi

def direction(grad):
    angle=grad/math.pi*180
    if between(angle,-157.5,-112.5) or between(angle,22.5,67.5):
        return -1,-1
    elif between(angle,-112.5,-67.5) or between(angle,67.5,112.5):
        return 0,-1
    elif between(angle,-67.5,-22.5) or between(angle,112.5,157.5):
        return 1,-1
    else:
        return -1,0

def canny(data):
    n,m=len(data),len(data[0])
    grad=[[0]*m for x in xrange(n)]
    tmp=[[0]*m for x in xrange(n)]
    for i in xrange(1,n-1):
        for j in xrange(1,m-1):
            gx=data[i+1][j-1]+data[i+1][j+1]+data[i+1][j]-data[i-1][j-1]-data[i-1][j+1]-data[i-1][j]
            gy=data[i-1][j+1]+data[i+1][j+1]+data[i][j+1]-data[i-1][j-1]-data[i+1][j-1]-data[i][j-1]
            grad[i][j]=math.atan2(gy,gx)
            tmp[i][j]=abs(gx)+abs(gy)
    ret=[[0]*m for x in xrange(n)]
    for x in xrange(1,n-1):
        for y in xrange(1,m-1):
            dx,dy=direction(grad[i][j])
            x0,x1=x-dx,x+dx
            y0,y1=y-dy,y+dy
            if tmp[x][y]>=tmp[x0][y0] and tmp[x][y]>=tmp[x1][y1]:
                ret[x][y]=tmp[x][y]
    return ret

def work1(image_name):
    image=Image.open(image_name)
    n,m=image.size
    data=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            data[i][j]=image.getpixel((i,j))
    get_image(roberts(data)).save('roberts_'+image_name)
    get_image(prewitt(data)).save('prewitt_'+image_name)
    get_image(sobel(data)).save('sobel_'+image_name)
    get_image(marr_hildreth(data)).save('marr_hildreth_'+image_name)
    get_image(canny(data)).save('canny_'+image_name)

def thresholding(data,threshold):
    n,m=len(data),len(data[0])
    ret=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            if data[i][j]>=threshold:
                ret[i][j]=255
    return ret

def average(offset,histogram):
    tot=num=0
    for i in xrange(len(histogram)):
        tot+=(offset+i)*histogram[i]
        num+=histogram[i]
    return tot/num

def global_thresholding(histogram):
    T=average(0,histogram)
    while True:
        m1=average(0,histogram[:T])
        m2=average(T,histogram[T:])
        T0=(m1*sum(histogram[:T])+m2*sum(histogram[T:]))/sum(histogram)
        if T==T0:
            return T
        T=T0

def otsu_thresholding(histogram):
    num=sum(histogram)
    L=len(histogram)
    p=[0]*L
    for i in xrange(L):
        p[i]=float(histogram[i])/num
    P=[0]*L
    P[0]=p[0]
    for i in xrange(1,L):
        P[i]=P[i-1]+p[i]
    m=[0]*L
    for i in xrange(1,L):
        m[i]=m[i-1]+i*p[i]
    mg=m[-1]
    best=ret=None
    for i in xrange(L):
        if P[i]==0 or P[i]==1:
            continue
        sigma=(mg*P[i]-m[i])**2/P[i]/(1-P[i])
        if not best or sigma>best:
            best=sigma
            ret=i
    return ret

def work2(image_name):
    image=Image.open(image_name)
    n,m=image.size
    histogram=[0]*256
    data=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            data[i][j]=image.getpixel((i,j))
            histogram[data[i][j]]+=1
    global_threshold=global_thresholding(histogram)
    get_image(thresholding(data,global_threshold)).save('global_'+image_name)
    otsu_threshold=otsu_thresholding(histogram)
    get_image(thresholding(data,otsu_threshold)).save('otsu_'+image_name)

os.chdir('../data')
work1('building.jpg')
work2('polymersomes.jpg')
