import os
import Image
import matplotlib.pyplot as plt

def plot(histogram,image_name):
    plt.clf()
    plt.xlabel('Gray Value')
    plt.ylabel('Count')
    plt.axis([0,len(histogram),0,(max(histogram)/10000+1)*10000])
    for i in xrange(len(histogram)):
        plt.bar(i,histogram[i])
    plt.savefig('histogram_'+image_name)

def equalization(histogram):
    tot_size=sum(histogram)
    max_level=len(histogram)
    acc_sum=0
    change=[0]*max_level
    for i in xrange(max_level):
        acc_sum+=histogram[i]
        change[i]=round((max_level-1.0)*acc_sum/tot_size)
    return change

def work(image_name):
    image=Image.open(image_name)
    histogram=[0]*256
    for i in xrange(image.size[0]):
        for j in xrange(image.size[1]):
            histogram[image.getpixel((i,j))]+=1
    plot(histogram,image_name)
    change=equalization(histogram)
    histogram=[0]*256
    for i in xrange(image.size[0]):
        for j in xrange(image.size[1]):
            image.putpixel((i,j),change[image.getpixel((i,j))])
            histogram[image.getpixel((i,j))]+=1
    plot(histogram,'new_'+image_name)
    image.save('new_'+image_name)

os.chdir('../data')
work('Fig1.jpg')
work('Fig2.jpg')
