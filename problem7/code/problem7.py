import os
import Image
import scipy.fftpack

def dct(data):
    data=scipy.fftpack.dct(data,norm='ortho').T
    data=scipy.fftpack.dct(data,norm='ortho').T
    return data

def idct(data):
    data=scipy.fftpack.idct(data,norm='ortho').T
    data=scipy.fftpack.idct(data,norm='ortho').T
    return data

def get_image(data):
    n,m=len(data),len(data[0])
    image=Image.new('L',(n,m))
    for i in xrange(n):
        for j in xrange(m):
            image.putpixel((i,j),data[i][j])
    return image

def normalize(data):
    n,m=len(data),len(data[0])
    ret=[[0]*m for x in xrange(n)]
    mi=min(min(x) for x in data)
    ma=max(max(x) for x in data)-mi
    for i in xrange(n):
        for j in xrange(m):
            ret[i][j]=(data[i][j]-mi)*255.0/ma
    return ret

def zonal(image_name,mask_size):
    image=Image.open(image_name)
    n,m=image.size
    ret=[[0]*m for x in xrange(n)]
    for i in xrange(n/8):
        for j in xrange(m/8):
            data=[[0]*8 for x in xrange(8)]
            for u in xrange(8):
                for v in xrange(8):
                    data[u][v]=float(image.getpixel((i*8+u,j*8+v)))
            data=dct(data)
            for u in xrange(8):
                for v in xrange(8):
                    if u+v>mask_size:
                        data[u][v]=0
            data=idct(data)
            for u in xrange(8):
                for v in xrange(8):
                    ret[i*8+u][j*8+v]=data[u][v]
    delta=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            delta[i][j]=image.getpixel((i,j))-ret[i][j]
    get_image(ret).save('zonal_'+str(mask_size)+'_'+image_name)
    get_image(normalize(delta)).save('delta_zonal_'+str(mask_size)+'_'+image_name)

def threshold(image_name,mask_size):
    image=Image.open(image_name)
    n,m=image.size
    ret=[[0]*m for x in xrange(n)]
    for i in xrange(n/8):
        for j in xrange(m/8):
            data=[[0]*8 for x in xrange(8)]
            for u in xrange(8):
                for v in xrange(8):
                    data[u][v]=float(image.getpixel((i*8+u,j*8+v)))
            data=dct(data)
            tmp=[]
            for u in xrange(8):
                for v in xrange(8):
                    tmp.append((abs(data[u][v]),u,v))
            tmp.sort()
            for k in xrange(64-mask_size):
                data[tmp[k][1]][tmp[k][2]]=0
            data=idct(data)
            for u in xrange(8):
                for v in xrange(8):
                    ret[i*8+u][j*8+v]=data[u][v]
    delta=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            delta[i][j]=image.getpixel((i,j))-ret[i][j]
    get_image(ret).save('threshold_'+str(mask_size)+'_'+image_name)
    get_image(normalize(delta)).save('delta_threshold_'+str(mask_size)+'_'+image_name)

def transpose(data):
    n,m=len(data),len(data[0])
    ret=[[0]*n for x in xrange(m)]
    for i in xrange(n):
        for j in xrange(m):
            ret[j][i]=data[i][j]
    return ret

def dwt(data,h0,h1):
    n,m=len(data),len(h0)
    ret=[0]*n
    for i in xrange(0,n,2):
        for j in xrange(m):
            ret[i/2]+=data[i+j if i+j<n else i+j-n]*h0[j]
            ret[i/2+n/2]+=data[i+j if i+j<n else i+j-n]*h1[j]
    return ret

def idwt(data,g0,g1):
    n,m=len(data),len(g0)
    ret=[0]*n
    lo=[0]*n
    hi=[0]*n
    for i in xrange(n/2):
        lo[i*2]=data[i]
        hi[i*2]=data[i+n/2]
    for i in xrange(n):
        for j in xrange(m):
            tmp=lo[i+j if i+j<n else i+j-n]*g0[j]+hi[i+j if i+j<n else i+j-n]*g1[j]
            ret[i+m-1 if i+m-1<n else i+m-1-n]+=tmp
    return ret

def dwt2(data,h0,h1,level):
    n,m=len(data),len(data[0])
    for k in xrange(level):
        tmp=[[0]*m for x in xrange(n)]
        for i in xrange(n):
            for j in xrange(m):
                tmp[i][j]=data[i][j]
        tmp=transpose(tmp)
        for i in xrange(m):
            tmp[i]=dwt(tmp[i],h0,h1)
        tmp=transpose(tmp)
        for i in xrange(n):
            tmp[i]=dwt(tmp[i],h0,h1)
        for i in xrange(n):
            for j in xrange(m):
                data[i][j]=tmp[i][j]
        n/=2
        m/=2
    return data

def idwt2(data,g0,g1,level):
    n,m=len(data),len(data[0])
    for i in xrange(level-1):
        n/=2
        m/=2
    for k in xrange(level):
        tmp=[[0]*m for x in xrange(n)]
        for i in xrange(n):
            for j in xrange(m):
                tmp[i][j]=data[i][j]
        for i in xrange(n):
            tmp[i]=idwt(tmp[i],g0,g1)
        tmp=transpose(tmp)
        for i in xrange(m):
            tmp[i]=idwt(tmp[i],g0,g1)
        tmp=transpose(tmp)
        for i in xrange(n):
            for j in xrange(m):
                data[i][j]=tmp[i][j]
        n*=2
        m*=2
    return data

def transform(image_name,h0,h1,g0,g1,wavelet_name):
    image=Image.open(image_name)
    n,m=image.size
    data=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            data[i][j]=image.getpixel((i,j))
    data=dwt2(data,h0,h1,3)
    get_image(normalize(data)).save(wavelet_name+'_transform_'+image_name)
    for i in xrange(n):
        for j in xrange(m):
            if abs(data[i][j])<=1:
                data[i][j]=0
    data=idwt2(data,g0,g1,3)
    delta=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            delta[i][j]=abs(image.getpixel((i,j))-data[i][j])
    get_image(data).save(wavelet_name+'_'+image_name)
    get_image(normalize(delta)).save('delta_'+wavelet_name+'_'+image_name)


def haar(image_name):
    sqrt2=2**0.5
    h0=[1/sqrt2,1/sqrt2]
    h1=[1/sqrt2,-1/sqrt2]
    g0=[1/sqrt2,1/sqrt2]
    g1=[-1/sqrt2,1/sqrt2]
    transform(image_name,h0,h1,g0,g1,'haar')

def daubechies(image_name):
    g0=[0.23037781,0.71484657,0.63088076,-0.02798376,-0.18703481,0.03084138,0.03288301,-0.01059740]
    h0=[0]*8
    g1=[0]*8
    h1=[0]*8
    for i in xrange(8):
        h0[i]=g0[7-i]
        g1[i]=h0[i]*(-1)**i
        h1[i]=g0[i]*(-1)**(i+1) 
    transform(image_name,h0,h1,g0,g1,'daubechies')

def symlet(image_name):
    g0=[0.0322,-0.0126,-0.0992,0.2979,0.8037,0.4976,-0.0296,-0.0758]
    h0=[0]*8
    g1=[0]*8
    h1=[0]*8
    for i in xrange(8):
        h0[i]=g0[7-i]
        g1[i]=h0[i]*(-1)**i
        h1[i]=g0[i]*(-1)**(i+1) 
    transform(image_name,h0,h1,g0,g1,'symlet')

def cohen_daubechies_feauveau(image_name):
    h0=[0,0.0019,-0.0019,-0.017,0.0119,0.0497,-0.0773,-0.0941,0.4208,0.8259,0.4208,-0.0941,-0.0773,0.0497,0.0119,-0.017,-0.0019,0.0010]
    h1=[0,0,0,0.0144,-0.0145,-0.0787,0.0404,0.4178,-0.7589,0.4178,0.0404,-0.0787,-0.0145,0.0144,0,0,0,0]
    g0=[0]*18
    g1=[0]*18
    for i in xrange(18):
        g0[i]=h1[i]*(-1)**(i+1)
        g1[i]=h0[i]*(-1)**i
    transform(image_name,h0,h1,g0,g1,'cohen_daubechies_feauveau')

def work(image_name):
    for i in [1,3,5,7]:
        zonal(image_name,i)
    for i in [4,8,16,32]:
        threshold(image_name,i)
    haar(image_name)
    daubechies(image_name)
    symlet(image_name)
    cohen_daubechies_feauveau(image_name)
    
os.chdir('../data')
work('lenna.jpg')
