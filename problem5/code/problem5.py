import os
import random
import Image
import numpy

def get_image(data):
    n,m=len(data),len(data[0])
    image=Image.new('L',(n,m))
    for i in xrange(n):
        for j in xrange(m):
            image.putpixel((i,j),data[i][j])
    return image

def gauss(data,mean,variance):
    n,m=len(data),len(data[0])
    ret=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            ret[i][j]=data[i][j]+random.gauss(mean,variance**0.5)
    return ret

def normalize(data):
    n,m=len(data),len(data[0])
    mi=min(min(numpy.real(x)) for x in data)
    ma=max(max(numpy.real(x)) for x in data)-mi
    ret=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            ret[i][j]=(numpy.real(data[i][j])-mi)*255.0/ma
    return ret

def blur(data,a,b,T):
    n,m=len(data),len(data[0])
    tmp=numpy.fft.fft2(data)
    tmp=numpy.fft.fftshift(tmp)
    for u in xrange(n):
        for v in xrange(m):
            t=numpy.pi*((u-n/2)*a+(v-m/2)*b)
            if t==0:
                continue
            H=T/t*numpy.sin(t)*numpy.exp(-1J*t)
            tmp[u][v]*=H
    tmp=numpy.fft.ifftshift(tmp)
    tmp=numpy.fft.ifft2(tmp)
    return normalize(tmp)

def inverse(data,a,b,T):
    n,m=len(data),len(data[0])
    tmp=numpy.fft.fft2(data)
    tmp=numpy.fft.fftshift(tmp)
    for u in xrange(n):
        for v in xrange(m):
            t=numpy.pi*((u-n/2)*a+(v-m/2)*b)
            if t==0:
                continue
            H=T/t*numpy.sin(t)*numpy.exp(-1J*t)
            if abs(H)>1e-10:
                tmp[u][v]/=H
    tmp=numpy.fft.ifftshift(tmp)
    tmp=numpy.fft.ifft2(tmp)
    return normalize(tmp)

def wiener_deconvolution(data,a,b,T,K):
    n,m=len(data),len(data[0])
    tmp=numpy.fft.fft2(data)
    tmp=numpy.fft.fftshift(tmp)
    for u in xrange(n):
        for v in xrange(m):
            t=numpy.pi*((u-n/2)*a+(v-m/2)*b)
            if t==0:
                continue
            H=T/t*numpy.sin(t)*numpy.exp(-1J*t)
            if abs(H)>1e-10:
                H2=abs(H)**2
                tmp[u][v]/=(H2+K)/H2*H
    tmp=numpy.fft.ifftshift(tmp)
    tmp=numpy.fft.ifft2(tmp)
    return normalize(tmp)

def restoration(image_name,variance,K):
    image=Image.open(image_name)
    n,m=image.size
    data=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            data[i][j]=image.getpixel((i,j))
    blur_data=gauss(blur(data,0.1,0.1,1),0,variance)
    get_image(blur_data).save('blur_'+str(variance)+'_'+image_name)
    inverse_data=inverse(blur_data,0.1,0.1,1)
    get_image(inverse_data).save('inverse_'+str(variance)+'_'+image_name)
    wiener_deconvolution_data=wiener_deconvolution(blur_data,0.1,0.1,1,K)
    get_image(wiener_deconvolution_data).save('wiener_deconvolution_'+str(variance)+'_'+image_name)

def work(image_name):
    noises=[0,1,150,650]
    K=[0,0.001,0.01,0.03]
    for i in xrange(4):
        restoration(image_name,noises[i],K[i])

os.chdir('../data')
random.seed(19930131)
work('book_cover.jpg')
