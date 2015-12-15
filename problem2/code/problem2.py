import os
import Image

def normalize(data):
    n,m=len(data),len(data[0])
    mi=min(min(x) for x in data)
    ma=max(max(x) for x in data)
    ma-=mi
    ret=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            ret[i][j]=int(255.0*(data[i][j]-mi)/ma)
    return ret

def get_image(data):
    n,m=len(data),len(data[0])
    image=Image.new('L',(n,m))
    for i in xrange(n):
        for j in xrange(m):
            image.putpixel((i,j),data[i][j])
    return image

def laplacian(data):
    n,m=len(data),len(data[0])
    ret=[[0]*m for x in xrange(n)]
    dx=[0,0,1,1,1,-1,-1,-1]
    dy=[1,-1,1,0,-1,1,0,-1]
    for i in xrange(1,n-1):
        for j in xrange(1,m-1):
            for k in xrange(8):
                x=i+dx[k]
                y=j+dy[k]
                ret[i][j]+=data[i][j]-data[x][y]
    return ret

def plus(a,b):
    n,m=len(a),len(a[0])
    ret=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            ret[i][j]=a[i][j]+b[i][j]
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

def average(data):
    n,m=len(data),len(data[0])
    ret=[[0]*m for x in xrange(n)]
    d=[-2,-1,0,1,2]
    for i in xrange(2,n-2):
        for j in xrange(2,m-2):
            tot=0
            for x in xrange(5):
                for y in xrange(5):
                    tot+=data[i+d[x]][j+d[y]]
            ret[i][j]=int(tot/9.0)
    return ret

def multiply(a,b):
    n,m=len(a),len(a[0])
    ret=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            ret[i][j]=int(a[i][j]*b[i][j]/255.0)
    return ret

def gamma(data):
    n,m=len(data),len(data[0])
    ret=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            ret[i][j]=int(min(max(data[i][j],0),255)**0.5)
    return ret

def work(image_name):
    image=Image.open(image_name)
    n,m=image.size
    data=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            data[i][j]=image.getpixel((i,j))
    laplace_data=laplacian(data)
    get_image(normalize(laplace_data)).save('laplace_'+image_name)
    laplacian_data=plus(data,laplace_data)
    get_image(laplacian_data).save('laplacian_'+image_name)
    sobel_data=sobel(data)
    get_image(sobel_data).save('sobel_'+image_name)
    average_sobel_data=average(normalize(sobel_data))
    get_image(average_sobel_data).save('average_sobel_'+image_name)
    product_data=multiply(laplacian_data,average_sobel_data)
    get_image(product_data).save('product_'+image_name)
    mix_data=plus(data,product_data)
    get_image(mix_data).save('mix_'+image_name)
    gamma_data=normalize(gamma(mix_data))
    get_image(gamma_data).save('gamma_'+image_name)

os.chdir('../data')
work('skeleton_orig.jpg')
