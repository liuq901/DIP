import os
import random
import Image

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

def uniform(data,mean,variance):
    plus=mean*2
    minus=(variance*12)**0.5
    a=(plus-minus)/2
    b=(plus+minus)/2
    n,m=len(data),len(data[0])
    ret=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            ret[i][j]=data[i][j]+random.uniform(a,b)
    return ret


def pepper(data,prob):
    n,m=len(data),len(data[0])
    ret=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            ret[i][j]=0 if random.random()<=prob else data[i][j]
    return ret

def salt(data,prob):
    n,m=len(data),len(data[0])
    ret=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            ret[i][j]=255 if random.random()<=prob else data[i][j]
    return ret

def arithmetic_mean(data,order):
    n,m=len(data),len(data[0])
    ret=[[0]*m for x in xrange(n)]
    t=order/2
    for i in xrange(t,n-t):
        for j in xrange(t,m-t):
            tot=0
            for dx in xrange(-t,t+1):
                for dy in xrange(-t,t+1):
                    tot+=data[i+dx][j+dy]
            ret[i][j]=int(tot/order**2.0)
    return ret

def geometric_mean(data,order):
    n,m=len(data),len(data[0])
    ret=[[0]*m for x in xrange(n)]
    t=order/2
    for i in xrange(t,n-t):
        for j in xrange(t,m-t):
            tot=1
            for dx in xrange(-t,t+1):
                for dy in xrange(-t,t+1):
                    tot*=max(1,data[i+dx][j+dy])
            ret[i][j]=int(tot**(1.0/(order**2)))
    return ret

def contra_harmonic_mean(data,order,Q):
    n,m=len(data),len(data[0])
    ret=[[0]*m for x in xrange(n)]
    t=order/2
    for i in xrange(t,n-t):
        for j in xrange(t,m-t):
            s1=s2=0
            for dx in xrange(-t,t+1):
                for dy in xrange(-t,t+1):
                    tmp=max(1,data[i+dx][j+dy])
                    s1+=tmp**(Q+1)
                    s2+=tmp**Q
            ret[i][j]=int(s1/s2)
    return ret

def median(data,order):
    n,m=len(data),len(data[0])
    ret=[[0]*m for x in xrange(n)]
    t=order/2
    for i in xrange(t,n-t):
        for j in xrange(t,m-t):
            tmp=[]
            for dx in xrange(-t,t+1):
                for dy in xrange(-t,t+1):
                    tmp.append(data[i+dx][j+dy])
            ret[i][j]=sorted(tmp)[order**2/2]
    return ret

def minmax(data,order,func):
    n,m=len(data),len(data[0])
    ret=[[0]*m for x in xrange(n)]
    t=order/2
    for i in xrange(t,n-t):
        for j in xrange(t,m-t):
            tmp=[]
            for dx in xrange(-t,t+1):
                for dy in xrange(-t,t+1):
                    tmp.append(data[i+dx][j+dy])
            ret[i][j]=func(tmp)
    return ret

def alpha_trimmed_mean(data,order,d):
    n,m=len(data),len(data[0])
    ret=[[0]*m for x in xrange(n)]
    t=order/2
    for i in xrange(t,n-t):
        for j in xrange(t,m-t):
            tmp=[]
            for dx in xrange(-t,t+1):
                for dy in xrange(-t,t+1):
                    tmp.append(data[i+dx][j+dy])
            ret[i][j]=sum(sorted(tmp)[d:-d])/(order**2-2*d)
    return ret

def work(image_name):
    image=Image.open(image_name)
    n,m=image.size
    data=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            data[i][j]=image.getpixel((i,j))
    gauss_data=gauss(data,0,400)
    get_image(gauss_data).save('gauss_'+image_name)
    get_image(arithmetic_mean(gauss_data,3)).save('arithmetic_mean_gauss_'+image_name)
    get_image(geometric_mean(gauss_data,3)).save('geometric_mean_gauss_'+image_name)
    pepper_data=pepper(data,0.1)
    salt_data=salt(data,0.1)
    get_image(pepper_data).save('pepper_'+image_name)
    get_image(salt_data).save('salt_'+image_name)
    get_image(contra_harmonic_mean(pepper_data,3,1.5)).save('contra_harmonic_mean_pepper_'+image_name)
    get_image(contra_harmonic_mean(salt_data,3,-1.5)).save('contra_harmonic_mean_salt_'+image_name)
    pepper_salt_data=pepper(salt(data,0.1),0.1)
    get_image(pepper_salt_data).save('pepper_salt_'+image_name)
    get_image(median(pepper_salt_data,3)).save('median_pepper_salt_'+image_name)
    get_image(median(median(pepper_salt_data,3),3)).save('median_median_pepper_salt_'+image_name)
    get_image(median(median(median(pepper_salt_data,3),3),3)).save('median_median_median_pepper_salt_'+image_name)
    get_image(minmax(pepper_data,3,max)).save('max_pepper_'+image_name)
    get_image(minmax(salt_data,3,min)).save('min_salt_'+image_name)
    uniform_data=uniform(data,0,800)
    get_image(uniform_data).save('uniform_'+image_name)
    uniform_pepper_salt_data=pepper(salt(uniform_data,0.1),0.1)
    get_image(uniform_pepper_salt_data).save('uniform_pepper_salt_'+image_name)
    get_image(arithmetic_mean(uniform_pepper_salt_data,5)).save('arithmetic_mean_uniform_pepper_salt_'+image_name)
    get_image(geometric_mean(uniform_pepper_salt_data,5)).save('geometric_mean_uniform_pepper_salt_'+image_name)
    get_image(median(uniform_pepper_salt_data,5)).save('median_uniform_pepper_salt_'+image_name)
    get_image(alpha_trimmed_mean(uniform_pepper_salt_data,5,5)).save('alpha_trimmed_mean_uniform_pepper_salt_'+image_name)

os.chdir('../data')
random.seed(19930131)
work('Circuit.jpg')
