Python 2.7.10 (default, May 23 2015, 09:40:32) [MSC v.1500 32 bit (Intel)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> Source code
***********Image show*************
import numpy as np
import cv2
kernel=np.ones((5,5),np.uint8)
img = cv2.imread('C:/Python27/nf.png',0)
cv2.imshow('img',img)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow('opening',opening)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closing',closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

*********erosion dilation******************
import numpy as np
import cv2
kernel=np.ones((5,5),np.uint8)
img = cv2.imread('C:/Python27/ak1.jpg',0)
img_erosion=cv2.erode(img,kernel,iterations=1)
img_dilation=cv2.dilate(img,kernel,iterations=1)
cv2.imshow('img',img)
cv2.imshow('erosion',img_erosion)
cv2.imshow('dilation',img_dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()
***********filtering***************
import numpy as np
import cv2
kernel=np.ones((5,5),np.uint8)
img = cv2.imread('1a.jpg')
cv2.imshow('img',img)
filter=cv2.Canny(img,100,200)
cv2.imwrite('C:/Python27/filter.jpg',filter)
cv2.imshow('Laplacian filter',filter)
cv2.waitKey(0)
cv2.destroyAllWindows()
************edge************
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('C:/Python27/1.jpg',0)
cv2.imshow('img',img)
edges = cv2.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

************normalization*************
import numpy as np
import cv2
img = cv2.imread('C:/Python27/1.jpg',0)
cv2.imshow('img',img)
norm_image=cv2.normalize(img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
print "n",norm_image
cv2.imshow('normalizedImg',norm_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

**********size*************
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('C:/Python27/1.jpg',0)
ret,thresh = cv2.threshold(img,0,230,cv2.THRESH_BINARY)
height, width = img.shape
print "height and width : ",height, width
size = img.size
aspect_ratio=float(width/height)
print "size of the image in number of pixels is ", size
print "aspect_ratio",aspect_ratio
#imgplot = plt.imshow(img, 'gray')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

***********features***********
import numpy as np
import cv2
img = cv2.imread('akLL.jpg',0)
cv2.imshow('img',img)
height, width = img.shape
print "height and width : ",height,width
size = img.size
aspect_ratio=float(width)/height
black=(width*height)-cv2.countNonZero(img)
white=np.sum(img==255)
ret,thresh=cv2.threshold(img,127,255,0)
contours,hierarchy=cv2.findContours(thresh,1,2)
cnt=contours[0]
area=cv2.contourArea(cnt)
mean,eigvec = cv2.PCACompute(img)
norm_image=cv2.normalize(img,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
print "size of image in number of pixels", size
print "aspect ratio", aspect_ratio
print "black pixels", black
print "white pixels", white
print "Area", area
print "n", norm_image
print "mean,eigvec", mean,eigvec
cv2.waitKey(0)
cv2.destroyAllWindows()


********Histogram***************
from matplotlib import pyplot as plt
import numpy as np
import cv2
z=cv2.imread('C:/Python27/flt5a.jpg',0)
z=np.random.normal(size=100)
vert_hist=np.histogram(z,bins=10)
ax1=plt.subplot(2,1,1)
ax1.plot(vert_hist[0],vert_hist[1][:-1],'*g')
ax2=plt.subplot(2,1,2)
ax2.hist(z,bins=10,orientation="vertical");
plt.show()
count, division = np.histogram(z)
print "values", count,division
x = np.mean(count)
print "count values", x
y = np.mean(division)
print "division values", y
cv2.waitKey()
cv2.destroyAllWindows()


******filters*************
import numpy as np
import cv2
from matplotlib import pyplot as plt
kernel=np.ones((5,5),np.uint8)
img = cv2.imread('C:/Python27/ah1.jpg',0)
cv2.imshow('img',img)
binary = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY ,41,3) 
cv2.imwrite('C:/Python27/binary.jpg',0)
cv2.imshow('binary',binary)
def inverte(img, name):
    img = (255-img)
    cv2.imwrite(name, img)
    complement = cv2.bitwise_not(img)
cv2.imwrite('C:/Python27/complement.jpg',0)
cv2.imshow('complement',img)
filter = cv2.medianBlur(img, 3)
cv2.imwrite('C:/Python27/filter.jpg',0)
cv2.imshow('Medianfilter',filter)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('C:/Python27/closing.jpg',0)
cv2.imshow('closing',closing)
edges = cv2.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
cv2.imwrite('C:/Python27/edges.jpg',0)
cv2.imshow('edges',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

**********statistical approach******
import xlrd  # install xlrd from              http://pypi.python.org/pypi/xlrd
wb = xlrd.open_workbook("C:/Users/Admin/Desktop/Book2.xlsx")       # xls file to read from
sh1 = wb.sheet_by_index(0)                   # first sheet in workbook
#sh2 = wb.sheet_by_name("abc
print "content of", sh1.name # name of sheet
for rownum in range(sh1.nrows): # sh1.nrows -> number of rows (ncols -> num columns) 
    print sh1.row_values(rownum)
x = 511
y = 769
col = sh1.col_values(0)              # column 0 as a list of string or numbers
print '"A" column content:'          # python index 0, 1.colunm, called A 
for cell in col: 
 if x <= cell <=y:
        print cell
    else:
        print "f"
a = 12
b = 14
col = sh1.col_values(1)               # column 0 as a list of string or numbers
print '"B" column content:'                           # python index 0, 1.colunm, called A 
for cell in col: 
    if a <= cell <=b:
        print cell
    else:
        print "f"
p = 30000
q = 60000
col = sh1.col_values(2)  # column 0 as a list of string or numbers
print '"C" column content:' # python index 0, 1.colunm, called A 
for cell in col: 
    if p <= cell <=q:
        print cell
    else:
        print "f"

r = 3000
s = 3900
col = sh1.col_values(3)  # column 0 as a list of string or numbers
print '"D" column content:' # python index 0, 1.colunm, called A 
for cell in col: 
    if r <= cell <=s:
        print cell
    else:
        print "f"
c = 55
d = 60
col = sh1.col_values(4)  # column 0 as a list of string or numbers
print '"E" column content:' # python index 0, 1.colunm, called A 
for cell in col: 
    if c <= cell <=d:
        print cell
    else:
        print "f"
#print sh1.col_values(1)


*********Delaunay************
import cv2.cv as cv
import random
def draw_subdiv_point( img, fp, color ):
    cv.Circle( img, (cv.Round(fp[0]), cv.Round(fp[1])), 3, color, cv.CV_FILLED, 8, 0 );
def draw_subdiv_edge( img, edge, color ):
    org_pt = cv.Subdiv2DEdgeOrg(edge);
    dst_pt = cv.Subdiv2DEdgeDst(edge);
    if org_pt and dst_pt :
 org = org_pt.pt;
        dst = dst_pt.pt;
        iorg = ( cv.Round( org[0] ), cv.Round( org[1] ));
        idst = ( cv.Round( dst[0] ), cv.Round( dst[1] ));
     cv.Line( img, iorg, idst, color, 1, cv.CV_AA, 0 );
def draw_subdiv( img, subdiv, delaunay_color, voronoi_color ):
    for edge in subdiv.edges:
        edge_rot = cv.Subdiv2DRotateEdge( edge, 1 )
        draw_subdiv_edge( img, edge_rot, voronoi_color );
        draw_subdiv_edge( img, edge, delaunay_color );
def locate_point( subdiv, fp, img, active_color ):
    (res, e0) = cv.Subdiv2DLocate( subdiv, fp );
    if res in [ cv.CV_PTLOC_INSIDE, cv.CV_PTLOC_ON_EDGE ]:
        e = e0
        while True:
            draw_subdiv_edge( img, e, active_color );
            e = cv.Subdiv2DGetEdge(e, cv.CV_NEXT_AROUND_LEFT);
            if e == e0:
                break
    draw_subdiv_point( img, fp, active_color );


def draw_subdiv_facet( img, edge ):
    t = edge;
    count = 0;
    # count number of edges in facet
    while count == 0 or t != edge:
        count+=1
        t = cv.Subdiv2DGetEdge( t, cv.CV_NEXT_AROUND_LEFT );
    buf = []
    # gather points
    t = edge;
    for i in range(count):
        assert t>4
        pt = cv.Subdiv2DEdgeOrg( t );
        if not pt:
            break;
        buf.append( ( cv.Round(pt.pt[0]), cv.Round(pt.pt[1]) ) );
        t = cv.Subdiv2DGetEdge( t, cv.CV_NEXT_AROUND_LEFT );
    if( len(buf)==count ):
        pt = cv.Subdiv2DEdgeDst( cv.Subdiv2DRotateEdge( edge, 1 ));
        cv.FillConvexPoly( img, buf, cv.RGB(random.randrange(256),random.randrange(256),random.randrange(256)), cv.CV_AA, 0 );
        cv.PolyLine( img, [buf], 1, cv.RGB(0,0,0), 1, cv.CV_AA, 0);
        draw_subdiv_point( img, pt.pt, cv.RGB(0,0,0));
def paint_voronoi( subdiv, img ):
    cv.CalcSubdivVoronoi2D( subdiv );
    for edge in subdiv.edges:
        # left
        draw_subdiv_facet( img, cv.Subdiv2DRotateEdge( edge, 1 ));
       # right
        draw_subdiv_facet( img, cv.Subdiv2DRotateEdge( edge, 3 ));
if __name__ == '__main__':
    win = "sc1a.jpg";
    rect = ( 0, 0, 600, 600 );
    active_facet_color = cv.RGB( 255, 0, 0 );
    delaunay_color  = cv.RGB( 0,0,0);
    voronoi_color = cv.RGB(0, 180, 0);
    bkgnd_color = cv.RGB(255,255,255);
    img = cv.CreateImage( (rect[2],rect[3]), 8, 3 );
    cv.Set( img, bkgnd_color );
    cv.NamedWindow( win, 1 );
    storage = cv.CreateMemStorage(0);
    subdiv = cv.CreateSubdivDelaunay2D( rect, storage );
    print "Delaunay triangulation will be build now interactively."
    print "To stop the process, press any key\n";
    for i in range(200):
        fp = ( random.random()*(rect[2]-10)+5, random.random()*(rect[3]-10)+5 )
        locate_point( subdiv, fp, img, active_facet_color );
        cv.ShowImage( win, img );
        if( cv.WaitKey( 50 ) >= 0 ):
            break;
        cv.SubdivDelaunay2DInsert( subdiv, fp );
        cv.CalcSubdivVoronoi2D( subdiv );
        cv.Set( img, bkgnd_color );
        draw_subdiv( img, subdiv, delaunay_color, voronoi_color );
        cv.ShowImage( win, img );
        if( cv.WaitKey( 50 ) >= 0 ):
            break;
    cv.Set( img, bkgnd_color );
    paint_voronoi( subdiv, img );
    cv.ShowImage( win, img );
    cv.WaitKey(0);
    cv.DestroyWindow( win );

---------------Logistic Regression---------------------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import cm
import os
def costFunction(theta, X, Y):
    m=len(Y)
    J = 0
    m,n=np.shape(X)
    T = np.matmul(np.transpose(theta),np.transpose(X))
    TT = np.exp(-T)
    x = TT + 1
    h = 1/x
    C = -Y * np.log(h) - (1  - Y) * np.log(1 - h)
    J = (1 / m) * np.sum(C)
    return J
def gradient(theta, X, Y):
    m=len(Y)
    J = 0
    p=1500
    J_history=np.zeros((p,1))
    grad = np.zeros((p, 3))
    alpha = 0.0001
    for i in range(0,p):
        grad[i][0]=theta0 = theta[0]
        grad[i][1]=theta1 = theta[1]
        grad[i][2]=theta2 = theta[2]
        J_history[i] = costFunction(theta, X, Y)
        m, n = np.shape(X)
        for iter in range(0,m):
            T = np.matmul(np.transpose(theta), np.transpose(X))
            TT = np.exp(-T)
            x = TT + 1
            h = 1 / x
                        J1 = h - Y
        
            theta00 = theta0 - alpha  * np.sum(J1 *X[:,0])
            theta11 = theta1 - alpha  * np.sum(J1 * X[:,1])
            theta22 = theta2 - alpha  * np.sum(J1 * X[:,2])
            theta0=theta00
            theta1=theta11
            theta2=theta22
            theta[0] = theta0
            theta[1] = theta1
            theta[2] = theta2
return  J_history,  grad, theta
def sigmoid(z):
#SIGMOID Compute sigmoid functoon
    print('z in sigmoid function...',z)
    g=np.exp(-z)+1
    g=1/g
    if(g<0.5):
        return 0
    else:
        return 1
## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.
print('load the file...')
X1,X2,Y = np.loadtxt('ex2data1.csv',delimiter=",", unpack=True)

m=len(X1)
print('m=',m)
X1=np.array(X1)
X2=np.array(X2)
X1=np.reshape(X1,(m,1))
X2=np.reshape(X2,(m,1))
X=np.append(X1,X2,axis=1)
m,n=X.shape
print('n=',n)
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
z=np.ones((m,1))
X=np.append(z,X,axis=1)
# Initialize fitting parameters
initial_theta =np.zeros((n + 1, 1))
# Compute and display initial cost and gradient
J = costFunction(initial_theta, X, Y)
print('J(theta)-=***   \n', J)
J_history,theta,theta=gradient(initial_theta, X, Y)
print(' theta values): \n')
print(theta)
I=np.array([1,20,25])
I=np.reshape(I,(1,3))
print('I=')
print(I)
print('Theta...')
print(theta)
z=np.matmul(I,theta)
prob = sigmoid(z)
print()
I=np.delete(I,0,1)
print(I)
print(' p ', prob)

-----------------Linear classifier---------------
import numpy as np
import matplotlib.pyplot as plt
def computeCost(X, Y, theta):
#   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
#   parameter for linear regression to fit the data points in X and y
# Initialize some useful values
    m1 = len(Y) # number of training examples
    J = 0
    h=np.matmul(theta,np.transpose(X))
    j=np.subtract(h, Y)
    J=(np.square(j))
    J=np.sum(J)
    J=J/(2*m1)
    return J
def gradientDescent(X, Y, theta, alpha, num_iters, X1):
 [theta, J_history]
#  GRADIENTDESCENT Performs gradient descent to learn theta
#   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta
#   by taking num_iters gradient steps with learning rate alpha
# Initialize some useful values
 m = len(Y) # number of training examples
    J_history = np.zeros(num_iters)
    T0=np.zeros(num_iters)
    T0[0]=0
    T1=np.zeros(num_iters)
    T1[0]=0
    for iter  in range(0, num_iters):
        T0[iter]=theta1=theta[0]
        T1[iter]=theta2=theta[1]
        h = np.matmul(theta, np.transpose(X))
        #h = np.dot(theta, np.transpose(X))
        J=np.subtract(h, Y)
        theta1=theta1-alpha/m*np.sum(J)
        theta2=theta2- (alpha/m)*np.sum(np.matmul(J,X1))
        theta[0] = theta1
        theta[1] = theta2
       J_history[iter] = computeCost(X, Y, theta)
    return theta, J_history, T0, T1
