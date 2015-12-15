import os
import Image
import matplotlib.pyplot as plt
from collections import deque

def white(image,x,y):
    return image.getpixel((x,y))>=210

def count(image,x0,y0):
    dx=[-1,0,1]
    dy=[-1,0,1]
    ret=0
    for i in xrange(3):
        for j in xrange(3):
            x=x0+dx[i]
            y=y0+dy[j]
            if x>=0 and x<image.size[0] and y>=0 and y<image.size[1] and white(image,x,y):
                ret+=1
    return ret

def erode(image):
    n,m=image.size
    tmp=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            if count(image,i,j)==9:
                tmp[i][j]=255
    for i in xrange(n):
        for j in xrange(m):
            image.putpixel((i,j),tmp[i][j])
    return image

def dilate(image):
    n,m=image.size
    tmp=[[0]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            if count(image,i,j)!=0:
                tmp[i][j]=255
    for i in xrange(n):
        for j in xrange(m):
            image.putpixel((i,j),tmp[i][j])
    return image

def erosion(image_name):
    erode(Image.open(image_name)).save('erosion_'+image_name)

def dilation(image_name):
    dilate(Image.open(image_name)).save('dilation_'+image_name)

def opening(image_name):
    dilate(erode(Image.open(image_name))).save('opening_'+image_name)

def closing(image_name):
    erode(dilate(Image.open(image_name))).save('closing_'+image_name)

def work(image_name):
    erosion(image_name)
    dilation(image_name)
    opening(image_name)
    closing(image_name)

def boundary_extraction(image_name):
    image=Image.open(image_name)
    tmp=erode(image.copy())
    n,m=image.size
    for i in xrange(n):
        for j in xrange(m):
            if white(image,i,j) and white(tmp,i,j):
                image.putpixel((i,j),0)
    image.save('boundary_extraction_'+image_name)

def hole(x):
    return x<=1000

def bfs(image,vis,x0,y0,func):
    dx=[-1,1,0,0]
    dy=[0,0,-1,1]
    queue=deque([(x0,y0)])
    vis[x0][y0]=True
    ret=1
    while queue:
        x0,y0=queue.popleft()
        for i in xrange(4):
            x=x0+dx[i]
            y=y0+dy[i]
            if x>=0 and x<image.size[0] and y>=0 and y<image.size[1] and not vis[x][y] and func(image,x,y):
                vis[x][y]=True
                ret+=1
                queue.append((x,y))
    return ret

def fill(image,x0,y0):
    dx=[-1,1,0,0]
    dy=[0,0,-1,1]
    queue=deque([(x0,y0)])
    while queue:
        x0,y0=queue.popleft()
        for i in xrange(4):
            x=x0+dx[i]
            y=y0+dy[i]
            if x>=0 and x<image.size[0] and y>=0 and y<image.size[1] and not white(image,x,y):
                image.putpixel((x,y),255)
                queue.append((x,y))

def hole_filling(image_name):
    image=Image.open(image_name)
    n,m=image.size
    vis=[[False]*m for x in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            if not vis[i][j] and not white(image,i,j) and hole(bfs(image,vis,i,j,lambda x,y,z:not white(x,y,z))):
                fill(image,i,j)
    image.save('hole_filling_'+image_name)

def plot(num,image_name):
    plt.clf()
    plt.xlabel('Connected Component')
    plt.ylabel('Pixel Number')
    n=len(num)
    plt.xticks(xrange(n),xrange(n))
    for i in xrange(n):
        plt.bar(i,num[i],align='center')
        length=len(str(num[i]))
        plt.text(i-length*0.1,num[i]+10,num[i])
    plt.savefig('connected_component_'+image_name)

def connected_component_extraction(image_name):
    image=Image.open(image_name)
    n,m=image.size
    for i in xrange(n):
        for j in xrange(m):
            if not white(image,i,j):
                image.putpixel((i,j),0)
    image=erode(image)
    vis=[[False]*m for x in xrange(n)]
    num=[]
    for i in xrange(n):
        for j in xrange(m):
            if not vis[i][j] and white(image,i,j):
                num.append(bfs(image,vis,i,j,white))
    image.save('connected_component_extraction_'+image_name)
    plot(num,image_name)

os.chdir('../data')
work('noisy_fingerprint.jpg')
boundary_extraction('licoln_from_penny.jpg')
hole_filling('region_filling_reflections.jpg')
connected_component_extraction('chickenfilet_with_bones.jpg')
