import os
import Image
import numpy

def get_image(f,M,N):
    image=Image.new('L',(M,N))
    for u in xrange(M):
        for v in xrange(N):
            image.putpixel((u,v),complex(f[u][v]).real)
    return image

def filtering(image_name,H,save_image_name):
    image=Image.open(image_name)
    M=image.size[0]
    N=image.size[1]
    P=M*2
    Q=N*2
    f=[[0.0]*Q for x in xrange(P)]
    for u in xrange(M):
        for v in xrange(N):
            f[u][v]=image.getpixel((u,v))
            if (u+v)%2==1:  
                f[u][v]*=-1
    F=numpy.fft.fft2(f)
    for u in xrange(P):
        for v in xrange(Q):
            F[u][v]*=H[u][v]
    f=numpy.fft.ifft2(F)
    for u in xrange(M):
        for v in xrange(N):
            if (u+v)%2==1:
                f[u][v]*=-1
    get_image(f,M,N).save('../data/'+save_image_name)

def D(u,v,P,Q):
    return ((u-P/2)**2+(v-Q/2)**2)**0.5

def ideal(M,N,D0,highpass):
    P=M*2
    Q=N*2
    H=[[0.0]*Q for x in xrange(P)]
    for u in xrange(P):
        for v in xrange(Q):
            if D(u,v,P,Q)<=D0:
                H[u][v]=1
            else:
                H[u][v]=0
            if highpass:
                H[u][v]=1-H[u][v]
    return H

def butterworth(M,N,D0,highpass):
    P=M*2
    Q=N*2
    H=[[0.0]*Q for x in xrange(P)]
    n=2
    for u in xrange(P):
        for v in xrange(Q):
            H[u][v]=1/(1+(D(u,v,P,Q)/D0)**(2*n))
            if highpass:
                H[u][v]=1-H[u][v]
    return H

def gaussian(M,N,D0,highpass):
    P=M*2
    Q=N*2
    H=[[0.0]*Q for x in xrange(P)]
    for u in xrange(P):
        for v in xrange(Q):
            H[u][v]=numpy.exp(-D(u,v,P,Q)**2/(2*D0**2))
            if highpass:
                H[u][v]=1-H[u][v]
    return H


def work(image_name):
    M,N=Image.open(image_name).size
    filters=[ideal,butterworth,gaussian]
    prefix=['ideal','butterworth','gaussian']
    for i in xrange(3):
        for D0 in [10,30,60,160,460]:
            H=filters[i](M,N,D0,False)
            filtering(image_name,H,prefix[i]+'_lowpass_'+str(D0)+'_'+image_name)
    for i in xrange(3):
        for D0 in [30,60,160]:
            H=filters[i](M,N,D0,True)
            filtering(image_name,H,prefix[i]+'_highpass_'+str(D0)+'_'+image_name)

os.chdir('../data')
work('characters_test_pattern.jpg')
