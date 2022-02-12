import numpy
import PIL.Image
import pickle
import xlwt

#输入图像预处理

def pretreatment(ima):
    ima=ima.convert('L')         #转化为灰度图像
    im=numpy.array(ima)        #转化为二维数组
    for i in range(im.shape[0]):#转化为二值矩阵
        for j in range(im.shape[1]):
            if im[i,j]==75 or im[i,j]==76:
                im[i,j]=1
            else:
                im[i,j]=0
    return im

#提取图片特征
def feature(A):  
    midx=int(A.shape[1]/2)+1
    midy=int(A.shape[0]/2)+1
    A1=A[0:midy,0:midx].mean()
    A2=A[midy:A.shape[0],0:midx].mean()
    A3=A[0:midy,midx:A.shape[1]].mean()
    A4=A[midy:A.shape[0],midx:A.shape[1]].mean()
    A5=A.mean()
    AF=[A1,A2,A3,A4,A5]
    return AF

#训练已知图片的特征         
def training():
    train_set={}
    for i in range(1,10):
        value=[]
        for j in range(1,6):
            ima=PIL.Image.open('F://DataStructure_SummerProject_C++//knn//data//'+str(i)+str(j)+'.jpg')
            im=pretreatment(ima)
            AF=feature(im)
            value.append(AF)
        train_set[i]=value
    #把训练结果存为永久文件，以备下次使用
    output=open('F://DataStructure_SummerProject_C++//knn//train_set.pkl','wb')
    pickle.dump(train_set,output)
    output.close()
    return train_set
    
#计算两向量的距离
def distance(v1,v2):
    vector1=numpy.array(v1)
    vector2=numpy.array(v2) 
    Vector=(vector1-vector2)**2
    distance = Vector.sum()**0.5
    return distance

#用最近邻算法识别国旗
def knn(train_set,V,k):
    key_sort=[9]*k
    value_sort=[9]*k
    for key in range(1,10):
        for value in train_set[key]:
            d=distance(V,value)
            for i in range(k):
                if d<value_sort[i]:
                    for j in range(k-2,i-1,-1):
                        key_sort[j+1]=key_sort[j]
                        value_sort[j+1]=value_sort[j]
                    key_sort[i]=key
                    value_sort[i]=d
                    break
    max_key_count=-1
    key_set=set(key_sort)
    for key in key_set:
        if max_key_count<key_sort.count(key):
            max_key_count=key_sort.count(key)
            max_key=key
    return max_key

#用于演示的国旗名字输出
def country_num(a):
    if a==1 :
        return '日本'
    elif a==2:
        return '中国'
    elif a==3 :
        return '加拿大'    
    elif a==4 :
        return '澳大利亚'
    elif a==5 :
        return '德国'
    elif a==6 :
        return '法国'
    elif a==7 :
        return '英国'
    elif a==8 :
        return '美国'
    elif a==9:
        return '俄罗斯'

train_set=training()
cnt=0 #识别正确的图片个数
for i in range(1,10):
    ima=PIL.Image.open('F://DataStructure_SummerProject_C++//knn//test//'+str(i)+'.jpg')
    im=pretreatment(ima)      #预处理
    AF=feature(im)  #分割并提取图片
    result =knn (train_set,AF,13) #knn识别
    if(result) == i:
        cnt= cnt + 1

print(cnt/9)