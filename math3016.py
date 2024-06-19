import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

#question 1
print('Question 1')
def f1(x):
    return x**2-np.log(x)

def goldenSection(a,b,eps):
    start = a
    end = b
    t = 0.618
    k = 1
    xv = np.arange(a, b, (b - a)/100)
    
    plt.plot(xv, f1(xv))
    plt.show()
    
    x1 = b - t*(b - a)
    x2 = a + t*(b - a)
    while (abs(b-a) > eps):
        
        if f1(x2) > f1(x1):
            b = x2
            x2 = x1
            x1 = b - t*(b - a)
        else:
            a = x1
            x1 = x2
            x2 = a + t*(b - a)
            
        k += 1
        
        plt.plot(xv, f1(xv))
        plt.scatter([a, b], [f1(a), f1(b)], color = "black")
        plt.scatter([x1, x2], [f1(x1), f1(x2)], color = "red")
        plt.show()
    
    print('Minimum of this function between the interval of ' + str(start) + ' and ' + str(end) + ' with epsilon = ' + str(eps) + ' is ' + str((a+b)/2))

print('(i)')
goldenSection(0.2,1.5,0.5)
print('(ii)')
goldenSection(0.2,1.5,0.001)
print('Hence by minimizing epsilon, we can find a more accurate solution than the one found in part i')
print('=============================================================')
print('Question 2')

#question 2
def f2(x):
    return np.sin(x)-np.exp(x)

def gradf2(x):
    return np.cos(x)-np.exp(x)

def gradientDescentFSS2(x,n,a):
    startpoint = x
    xv = [f2(x)]
    for i in range(n-1):
        x = x - a*gradf2(x)
        xv.append(f2(x))
    plt.plot(range(n),xv)
    plt.show()
    
    print('Solution found by the gradient descent algorithm with fixed step size with start point ' + str(startpoint) + ', ' + str(n) + ' iterations and step size ' + str(a) + ' is ' + str(x))

print('(i)')
gradientDescentFSS2(-2,100,0.1)

print('(ii)')
gradientDescentFSS2(-2,100,10)
print('Hence due to high alpha, it diverges')

print('=============================================================')
print('Question 3')
#question 3

def mcCormick(x):
    return np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] +2.5 * x[1] + 1

def gradMcCormick(x):
    return np.array([np.cos(x[0]+x[1]) + 2 * x[0] - 2 * x[1] - 1.5, 
                     np.cos(x[0]+x[1]) - 2 * x[0] + 2 * x[1] + 2.5])

def gradientDescentFSS3(x,n,a):
    startpoint = x
    xv = [mcCormick(x)]
    for i in range(n-1):
        x = x - a*gradMcCormick(x)
        xv.append(mcCormick(x))
    plt.plot(range(n),xv)
    plt.show()
    
    print('Solution found by the gradient descent algorithm with fixed step size with start point ' + str(startpoint) + ', ' + str(n) + ' iterations and step size ' + str(a) + ' is ' + str(x) + '\nfunction value: ' + str(mcCormick(x)))


def accGradientDescent(x,n,a):
    startpoint = x
    xTmp = x
    xv = [mcCormick(x)]
    for i in range(1,n-1):
        y = x + i/(i+3)*(x-xTmp)
        xTmp = x
        x = y - a*gradMcCormick(y)
        xv.append(mcCormick(x))
    plt.plot(range(n-1),xv)
    plt.show()
        
    print('Solution found by the accelerated gradient descent algorithms with start point ' + str(startpoint) + ', ' + str(n) + ' iterations and step size ' + str(a) + ' is ' + str(x) + '\nfunction value: ' + str(mcCormick(x)))

print('(i)')
gradientDescentFSS3(np.array([-8,4]),400,0.001)
accGradientDescent(np.array([-8,4]),400,0.001)
print('In this case, gradient descent with fixed step size performs better as it gets a lower value than the other algorithm.')


def f4(x):
    return x[0] ** 4 + x[1] ** 4 - x[0] ** 2 - x[1] ** 2

def gradf4(x):
    return np.array([4 * x[0] ** 3 - 2 * x[0], 4 * x[1] **3 - 2 * x[1]])

def gradientDescentFSS4(x,n,a):
    startpoint = x
    xv = [f4(x)]
    for i in range(n-1):
        x = x - a*gradf4(x)
        xv.append(f4(x))
    plt.plot(range(n),xv)
    plt.show()
    
    print('Solution found by the gradient descent algorithm with fixed step size with start point ' + str(startpoint) + ', ' + str(n) + ' iterations and step size ' + str(a) + ' is ' + str(x) + '\nfunction value: ' + str(f4(x)))

def accGradientDescent4(x,n,a):
    startpoint = x
    xTmp = x
    xv = [f4(x)]
    for i in range(1,n-1):
        y = x + i/(i+3)*(x-xTmp)
        x = y - a*gradf4(y)
        xTmp = x
        xv.append(f4(x))
    plt.plot(range(n-1),xv)
    plt.show()
        
    print('Solution found by the accelerated gradient descent algorithms with start point ' + str(startpoint) + ', ' + str(n) + ' iterations and step size ' + str(a) + ' is ' + str(x) + '\nfunction value: ' + str(f4(x)))

print('(ii)')
gradientDescentFSS4(np.array([-1,1]),100,0.001)
accGradientDescent4(np.array([-1,1]),100,0.001)
print('In this case, gradient descent with fixed step size performs better as it gets a lower value than the other algorithm.')

print('(iii)')
print('Modify the starting point and iterations')
gradientDescentFSS4(np.array([1, -1]), 400, 0.001)
accGradientDescent4(np.array([1, -1]), 400, 0.001)
print('In this case, different starting points and increased iterations might lead to finding a different local minimum.')

print('=============================================================')
print('Question 4')

def bohachevsky(x):
    return x[0] ** 2 + x[1] ** 2 - 0.3 * np.cos(3 * np.pi * x[0]) - 0.4 * np.cos(4 * np.pi * x[1]) + 0.7

def gradBohachevsky(x):
    return np.array([2 * x[0] + 0.9 * np.pi * np.sin(3 * np.pi * x[0]),
                     2 * x[1] + 1.6 * np.pi * np.sin(4 * np.pi * x[1])])
    
def hasBohachevsky(x):
    return np.array([[2 + 2.7 * np.pi ** 2 * np.cos(3 * np.pi * x[0]), 0],
                     [0 ,2 + 6.4 * np.pi ** 2 * np.sin(4 * np.pi * x[1])]])
    
def newton(x,n):
    startPoint = x
    xv = [bohachevsky(x)]
    k = 1
    while (abs(gradBohachevsky(x)) > 10 ** (-8)).any() and k < n:
        x = x - gradBohachevsky(x)@np.linalg.inv(hasBohachevsky(x))
        k += 1
        xv.append(bohachevsky(x))
        
    plt.plot(range(k),xv)
    plt.show()
    
    print('Solution of Newton method of Bohachevsky function with start point ' + str(startPoint) + ' and '+ str(k) + ' iterations is ' + str(x))




def newtonLs(x,n):
    startPoint = x
    xv = [bohachevsky(x)]
    k = 1
    while (abs(gradBohachevsky(x)) > 10 ** (-8)).any() and k < n:
        d = - gradBohachevsky(x)@np.linalg.inv(hasBohachevsky(x))
        
        res = minimize_scalar(lambda alpha: bohachevsky(x + alpha*d))
        alpha = res.x
        
        x = x + alpha*d
        k += 1
        xv.append(bohachevsky(x))
        
    plt.plot(range(k),xv)
    plt.show()
    print('Solution of Newton method with line search of Bohachevsky function with start point ' + str(startPoint) + ' and '+ str(k) + ' iterations is ' + str(x))

newton(np.array([-5,2]),100)
newtonLs(np.array([-5,2]),100)
newton(np.array([-4.9,2.1]),100)
newtonLs(np.array([-4.9,2.1]),100)
newton(np.array([-10,1]),100)
newtonLs(np.array([-10,1]),100)
print('=============================================================')
print('Question 5')

def f5(x):
    return 3/2*x[0]**2-x[0]*x[1]+3/2*x[1]**2-x[1]+7

def linearConjugateGradient(A,b,x):
    startPoint = x
    k = 0
    r0 = A@x -b
    d = -r0
    alpha = -(r0@d)/(d@A@d)
    x = x + alpha * d
    r1 = A@x - b
    k += 1
    if (r1 != 0).any():
        r = r1
    else: 
        return x
    
    while (r != 0).any():
        beta = (r@A@d)/(d@A@d)
        d = -r + beta * d
        alpha = -(r@d)/(d@A@d)
        x = x + alpha * d
        r = A@x -b
        k += 1
        
    print('The solution of linear conjugate gradient algorithm with start point ' + str(startPoint) + ' matrix ' + str(A) + ' and vector ' + str(b) + ' is ' + str(x) + ' with ' + str(k) + ' iterations. ' + '\n The minimum of this function is '+ str(f5(x)))
        

A = np.array([[3,-1],[-1,3]])
b = np.array([0,1])
x = np.array([0,0])
linearConjugateGradient(A,b,x)
