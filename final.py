
# coding: utf-8

# In[3]:

import tkinter
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
import numpy as np
import math
import pylab
from scipy import optimize

from matplotlib.figure import Figure



# In[5]:

def get_points():
    root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("all files","*.*"), ("jpeg files","*.jpg")))
    print (root.filename)
    if len(root.filename):
        if True: #try:
            f = open(root.filename)
            #h = []
            array = []
            t = []
            for line in f:
                array.append(float(line.split()[1]))
                t.append(float(line.split()[0]))
            #h = np.array([t[i+1]-t[i] for i in range(len(t)-1)])
            #print(h)
            #if sum(h <= 0) > 1 or sum((h - h[0]) > 0.1) > 0:
            #    print(sum(h <= 0), sum((h - h[0]) > 0.1))
            #    array = []
            #    messagebox.showinfo("Error", "Error")
            #    return False
            #else:
            return True, array, t
        #except Exception:
        #    array = []
        #    messagebox.showinfo("Error", "Error")
        #    return False


# In[6]:

def get_filename_ro():
    global ro
    global ro_points
    global w
    f, ar, t = get_points()
    if f:
        ro = None
        ro_points = ar[:]
        w = t[:]

def get_filename_s():
    global S
    global s_points
    global t_s
    f, ar, t = get_points()
    if f:
        S = None
        s_points = ar[:]
        t_s = t[:]

def get_filename_z():
    global z
    global z_points
    global t_z
    f, ar, t = get_points()
    if f:
        z = None
        z_points = ar[:]
        t_z = t[:]

# In[7]:

def callback_ro():
    if True:#try:
        a = float(e1.get())
        b = float(e2.get())
        global ro 
        ro = (lambda w: a*w*(b-w))
        bt_ro['text'] = "Saved"
    #except Exception:
    #    messagebox.showinfo("Error", "Error")

def callback_S():
    try:
       a = float(e11.get())
       b = float(e22.get())
       global S 
       S = (lambda t: a*t + b*np.sin(t))
       bt_s['text'] = "Saved"
    except Exception:
        messagebox.showinfo("Error", "Error")

def callback_z():
    try:
       a = float(z_e1.get())
       b = float(z_e2.get())
       global z
       z = (lambda t: a*t + b*np.cos(t))
       bt_z['text'] = "Saved"
    except Exception:
        messagebox.showinfo("Error", "Error")
        
def callback_Bt():
    global Bt
    try:
        Bt = float(bt_en.get())
        print(Bt)
        b2['text'] = "Saved"
    except Exception:
        messagebox.showinfo("Error", "Error")
        
def callback_x2():
    global X0
    try:
        X0 = [float(x_e1.get()), float(x_e2.get())]
        bt_x['text'] = "Saved"
    except Exception:
        messagebox.showinfo("Error", "Error")

def callback_y2():
    global Y0
    try:
        Y0 = [float(xy1.get()), float(xy2.get())]
        bt_xy['text'] = "Saved"
    except Exception:
        messagebox.showinfo("Error", "Error")


def callback_x():
    global X0
    global Y0
    try:
        X0 = float(x_e1.get())
        Y0 = float(x_e2.get())
        print("X0:", X0, "Y0:", Y0)
        bt_x['text'] = "Saved"
    except Exception:
        messagebox.showinfo("Error", "Error")

def callback_Bt_T():
    global T
    try:
        T = float(bt_T1.get())
        if T <= 0:
            T = None
            messagebox.showinfo("Error", "Error")
        else:
            bt_T2['text'] = "Saved"
    except Exception:
        messagebox.showinfo("Error", "Error")

def callback_Bt_int():
    global Bt
    try:
        #Bt = [min(float(bt_en1.get()), float(bt_en2.get())), max(float(bt_en1.get()), float(bt_en2.get()))]
        Bt = [float(bt_en1.get()), float(bt_en2.get())]
        print(Bt)
        b2['text'] = "Saved"
    except Exception:
        messagebox.showinfo("Error", "Error")


# In[8]:

def sel():
    selection = "You selected the option " + str(var.get())
    label_sel.config(text = selection)
    global f
    if var.get() == 1:
        f = (lambda s, x, Bt: Bt)
    else:
        f = (lambda s, x, Bt: Bt*(s-x))


# In[9]:

def write_to_file(x, array, path):
    f = open(path, 'w')
    for i in range(len(array)):
        f.write(str(x[i])+" "+ str(array[i]) + '\n')

def write_to_file2(array, path):
    f = open(path, 'w')
    for a in array:
        f.write(str(a)+" ")


# ## Cодержательная часть

# In[10]:

def Tabulate(x, f, path = ""):
    print("Tabulate")
    array = f(x)
    if len(path):
        write_to_file(x, array, path)
    return array


# In[11]:

def Integrate(x0, f0, a, b):
    #print("Integrate")
    x = x0[a <= x0]
    x = x0[x0 <= b]
    f = f0[len(f0)-len(x):len(f0)]
    if len(x) < 2:
        return (f0[len(f0)-1] + f0[len(f0)-2])*(x0[1]-x0[0])/2
    sum_int = f[0]+f[-1]+2*sum(f[1:-1])
    return float(sum_int*(x[1]-x[0])/2)


# In[12]:

def TabulateIntegral(ro_points, w, path = ""):
    w = np.array(w)
    print("TabulateIntegral")
    y_tab = np.linspace(0, 0.9, 100)
    U_y = []
    #W = np.linspace(y, 0.999999, 50)
    for y in y_tab:
        w_y = w[y <= w]
        if len(w_y >= 2):
            U_y.append(Integrate(w_y, ro_points[len(ro_points)-len(w_y):len(ro_points)], y, 1)) #
            #U_y.append(Integrate(W, ro(W), y, 1))
    #print(len(y_tab), len(U_y))
    return y_tab, U_y


# In[13]:

def LinSys(A, b):
    m = len(b)-1
    Ar = [[-A[0][1]/A[0][0], b[0]/A[0][0]]]
    for i in range(1, m):
        denum = A[i][i]+A[i][i-1]*Ar[-1][0]
        Ar.append([-A[i][i+1]/denum,
                   (b[i]-A[i][i-1]*Ar[-1][1])/denum])
    denum = A[m][m]+A[m][m-1]*Ar[-1][0]
    ans = [(b[m]-A[m][m-1]*Ar[-1][1])/denum]
    for i in reversed(range(m)):
        ans.append(Ar[i][0]*ans[-1]+Ar[i][1])
    return list(reversed(ans))


# In[14]:

def Interp(x, f, path = ""):
    n = len(x)-1
    teta = [x[i]-x[i-1] for i in range(1, len(x))]
    A = np.zeros((n-1, n-1))
    for i in range(n-2):
        A[i][i] = (teta[i]+teta[i+1])/3
        A[i][i+1] = teta[i+1]/6
        A[i+1][i] = teta[i+1]/6
    A[n-2][n-2] = (teta[n-2]+teta[n-1])/3
    b = np.array([(f[i+1]-f[i])/teta[i] - (f[i]-f[i-1])/teta[i-1] for i in range(1, n)])
    m = np.hstack(([0], np.array(LinSys(A, b)), [0]))
    A = np.zeros(n)
    B = np.zeros(n)
    for i in range(n):
        A[i] = (f[i+1]-f[i])/teta[i] - teta[i]*(m[i+1]-m[i])/6
        B[i] = f[i]-m[i]*teta[i]**2/6 - A[i]*x[i]
    Splain = []
    for i in range(n):
        sp = np.zeros(4)
        sp[0] = (m[i+1]-m[i])/(6*teta[i])
        sp[1] = (m[i]*x[i+1] - m[i+1]*x[i]) / (2*teta[i])
        sp[2] = (m[i+1]*x[i]**2 - m[i]*x[i+1]**2) / (2*teta[i]) + A[i]
        sp[3] = (m[i]*x[i+1]**3 - m[i+1]*x[i]**3) / (6*teta[i]) + B[i]
        Splain.append(sp)
    return Splain


# In[15]:

def Diff(splain):
    #print("Diff")
    diff_splain = []
    for sp in splain:
        diff_splain.append([3*sp[0], 2*sp[1], sp[2]])
    return diff_splain


# In[16]:

def search(array, points, t):
    #print(len(array), len(points), t)
    i = 0
    #if t < 0:
    #    print(t)
    #    return
    while t > points[i]:
        if i == len(array)-1:
            break
        i += 1
    if len(array[i]) == 3:
        return array[i][0]*t**2 + array[i][1]*t + array[i][2]
    else:
        return array[i][0]*t**3 + array[i][1]*t**2 + array[i][2]*t +  array[i][3]


# In[17]:

def func1(t, x, y, diff_z, U_y_interp, t_points, y_points):
    #print(y)
    return search(diff_z, t_points, t)*search(U_y_interp, y_points, y)


# In[18]:

def func2(t, x, y, s_interp, t_points, bt):
    return f(search(s_interp, t_points, t), x, bt)   #Bt*(x-search(z_interp, t_points, t))


# In[114]:

def DiffEq(func1, func2, Y0, Z0,  diff_z, U_y_interp, t_s, t_z, y_points, s_interp, bt, T):
    mas_y = [Y0]
    mas_z = [Z0]
    h = T/15
    #h = 0.1
    #n = T*10
    for i in range(15):
        #print("h", i*h+h/2)
        k0 =func1(i*h, mas_y[i], mas_z[i], diff_z, U_y_interp, t_z, y_points)
        l0 =func2(i*h, mas_y[i], mas_z[i], s_interp, t_s, bt)
        k1 = func1(i*h+h/2, mas_y[i]+ h*k0/2, mas_z[i] + h*l0/2, diff_z, U_y_interp, t_z, y_points)
        l1 = func2(i*h+h/2, mas_y[i]+ h*k0/2, mas_z[i] + h*l0/2, s_interp, t_s, bt)
        k2 = func1(i*h+h/2, mas_y[i]+ h*k1/2, mas_z[i] + h*l1/2, diff_z, U_y_interp, t_z, y_points)
        l2 = func2(i*h+h/2, mas_y[i]+ h*k1/2, mas_z[i] + h*l1/2, s_interp, t_s, bt)
        k3 = func1(i*h+h, mas_y[i]+ h*k2, mas_z[i] + h*l2, diff_z, U_y_interp, t_z, y_points)
        l3 = func2(i*h+h, mas_y[i]+ h*k2, mas_z[i] + h*l2, s_interp, t_s, bt)
        mas_y.append(mas_y[-1] + h*(k0 + 2*k1 + 2*k2 + k3)/6)
        mas_z.append(mas_z[-1] + h*(l0 + 2*l1 + 2*l2 + l3)/6)
        #write_to_file2(array_x, "/home/asya/hell/x.txt")
        #write_to_file2(array_y, "/home/asya/hell/y.txt")
    return mas_y, mas_z, np.linspace(0, T, 16)


# In[93]:

def functionFBeta(X, s_points, X_0, ro_points, w, t_points, Y, T):
    #Integrate(ro_interp, 1, 2)
    C2 = np.abs(X[-1] - s_points[-1])/s_points[-1]
    x_interp = Interp(t_points, X)
    diff_x = Diff(x_interp)
    interp_y = Interp(t_points, Y)
    x_w = []
    for t in t_points:
        y = search(interp_y, t_points, t)
        w = np.array(w)
        ro_points = np.array(ro_points)
        i = Integrate(w, w*ro_points, y, 1)
        x_w.append(search(diff_x, t_points, t)*i)
        #print(x_w[-1])
    if X[-1] != X_0:
        C1 = 1 - Integrate(t_points, x_w, 0, T)/(X[-1] - X_0)
    #print(C1, C2)
    return C1, C2


# In[99]:

def BetaSearch(Bt, func1, func2, X_0, Y_0, diff_z, U_y_interp, t_s, t_z, y_points, 
               s_interp, s_points, ro_points, w, T):
        Bt_ar = np.linspace(Bt[0], Bt[1], 10)
        Mas = []
        Mas2 = []
        for B in Bt_ar:
            #try:
            X, Y, t_xy = DiffEq(func1, func2, X_0, Y_0, diff_z, U_y_interp, t_s, t_z, 
                                y_points, s_interp, B, T)
            C_1, C_2 = functionFBeta(X, s_points, X_0, ro_points, w, t_xy, Y, T)
            #except Exception:
            #    Mas.append([0, 0, np.inf, np.inf])
            #    Mas2.append(np.inf)
            #else:
            if sum(np.isnan(X)) == 0 and sum(np.isnan(Y)) == 0 and np.isnan(C_1+10*C_2) == False:
                Mas.append([X, Y, t_xy, B, C_1, C_2])
                Mas2.append(C_1+10*C_2)
        if len(Mas) != 0:
            print([np.min(Mas[i][1]) < 0 for i in range(len(Mas))])
            if np.sum([np.min(Mas[i][1]) < 0 for i in range(len(Mas))]) < len(Mas):
                Mas2 = [Mas2[i] + 10000*(np.min(Mas[i][1]) < 0) for i in range(len(Mas))]
                ind = np.argmin(Mas2)
                print("min", Mas2[ind])
                print(len(Mas[ind][1]))
                return Mas[ind]


# In[131]:
def draw_y(t, Y):
    wdw3 = Toplevel()
    wdw3.geometry('700x700')
    f3 = Figure(figsize=(10, 10), dpi=100)
    a = f3.add_axes([0.1, 0.1, 0.8, 0.8])
    a.plot(t, Y, 'b', label = r'$y(t)$')
    a.set_title('dynamics')
    a.set_xlabel('t')
    a.set_ylabel('y(t)')
    a.legend(loc = 0, fontsize = 18)
    canvas = FigureCanvasTkAgg(f3, master=wdw3)
    canvas.show()
    canvas.get_tk_widget().pack()

def draw_Sx(t, array):
    wdw4 = Toplevel()
    wdw4.geometry('700x700')
    f4 = Figure(figsize=(10, 10), dpi=100)
    a = f4.add_axes([0.1, 0.1, 0.8, 0.8])
    a.plot(t, array, 'b', label = r'$|S(t)-x(t)|$', color='black')
    a.set_title('dynamics')
    a.set_xlabel('t')
    a.set_ylabel('|S(t)-x(t)|')
    a.legend(loc = 0, fontsize = 18)
    canvas = FigureCanvasTkAgg(f4, master=wdw4)
    canvas.show()
    canvas.get_tk_widget().pack()

def draw_xS(t, X, S):
    wdw5 = Toplevel()
    wdw5.geometry('700x700')
    f5 = Figure(figsize=(10, 10), dpi=100)
    a = f5.add_axes([0.1, 0.1, 0.8, 0.8])
    a.plot(X, S, 'b', label = r'$S(x)$', color = 'red')
    #a.plot(t, S, 'b', label = r'$S$', color = 'green')
    a.set_title('dynamics')
    a.set_xlabel('x')
    a.set_ylabel('S(x)')
    a.legend(loc = 0, fontsize = 18)
    canvas = FigureCanvasTkAgg(f5, master=wdw5)
    canvas.show()
    canvas.get_tk_widget().pack()

def draw_S_x(t, X, S):
    wdw6 = Toplevel()
    wdw6.geometry('700x700')
    f6 = Figure(figsize=(10, 10), dpi=100)
    a = f6.add_axes([0.1, 0.1, 0.8, 0.8])
    a.plot(t, X, 'b', label = r'$X(t)$')
    a.plot(t, S, 'b', label = r'$S(t)$', color='red')
    a.set_title('dynamics')
    a.set_xlabel('t')
    a.set_ylabel('S(t)')
    a.legend(loc = 0, fontsize = 18)
    canvas = FigureCanvasTkAgg(f6, master=wdw6)
    canvas.show()
    canvas.get_tk_widget().pack()

def draw_F(t, ar):
    wdw7 = Toplevel()
    wdw7.geometry('700x700')
    f7 = Figure(figsize=(10, 10), dpi=100)
    a = f7.add_axes([0.1, 0.1, 0.8, 0.8])
    a.plot(t, ar, 'y', label = r'$C_1 + 10*C_2$')
    a.set_title('dynamics')
    a.set_xlabel('Beta')
    a.set_ylabel('F')
    a.legend(loc = 0, fontsize = 18)
    canvas = FigureCanvasTkAgg(f7, master=wdw7)
    canvas.show()
    canvas.get_tk_widget().pack()

def draw_ro(t, Ro):
    wdw1 = Toplevel()
    wdw1.geometry('700x700')
    f = Figure(figsize=(10, 10), dpi=100)
    a = f.add_axes([0.1, 0.1, 0.8, 0.8])
    a.plot(t, Ro, 'b', label = r'$ro(w)$')
    a.set_title('dynamics')
    a.set_xlabel('w')
    a.set_ylabel('ro')
    a.legend(loc = 0, fontsize = 18)
    canvas = FigureCanvasTkAgg(f, master=wdw1)
    canvas.show()
    canvas.get_tk_widget().pack()


def draw_z(t, array):
    wdw4 = Toplevel()
    wdw4.geometry('700x700')
    f4 = Figure(figsize=(10, 10), dpi=100)
    a = f4.add_axes([0.1, 0.1, 0.8, 0.8])
    a.plot(t, array, label = r'$Z(t)$')
    a.set_xlabel('t')
    a.set_ylabel('z(t)')
    a.legend(loc = 0, fontsize = 18)
    canvas = FigureCanvasTkAgg(f4, master=wdw4)
    canvas.show()
    canvas.get_tk_widget().pack()


def Draw_hand_mode(t, X, Y, S, ro_points, w, t_z, z_points):
    draw_xS(t, X, S)
    draw_y(t, Y)
    draw_Sx(t, np.abs(X-S))
    draw_z(t_z, z_points)
    draw_ro(w, ro_points)
    draw_S_x(t, X, S)

def Solver():
    if True:  #try:
        #t = np.arange(5)
        global ro_points
        global ro
        if ro != None:
            w = np.linspace(0, 1, 1000)
            ro_points = Tabulate(w, ro, "/home/asya/hell/ro.txt")
        else:
            print("from file")
        global s_points
        global S
        if S != None:
            t_s = np.linspace(0, T, int(T*10 + 7))
            s_points = Tabulate(t_s, S, "/home/asya/hell/S.txt")
        global z_points
        global z
        if z != None:
            t_z = np.linspace(0, T, int(T*10 + 7))
            z_points = Tabulate(t_z, z, "/home/asya/hell/z.txt")
        
        global w
        global t_s
        global t_z                     
        
        #print(ro_points)
        
        #print(U_y_points)
        
        ro_interp = Interp(w, ro_points) #w
        s_interp = Interp(t_s, s_points)
        z_interp = Interp(t_z, z_points)
        
        #print(ro_interp)
        y_points, U_y_points = TabulateIntegral(ro_points, w, "/home/asya/hell/U_y.txt")
        #y = np.array(y_points)

        
        U_y_interp = Interp(y_points, U_y_points)
        
        diff_z = Diff(z_interp)
        if result:
            X, Y, t_xy = DiffEq(func1, func2, X0, Y0, diff_z, U_y_interp, t_s, t_z, y_points, s_interp, Bt, T)
            write_to_file(t_xy, X, "/home/asya/hell/x.txt")
            write_to_file(t_xy, Y, "/home/asya/hell/y.txt")
            print(Y)
            if sum(np.isnan(Y)) > 0:
               print("******************************************************************")
               print("В процессе вычисления произошло переполнение")
            else:
                #if S != None:
                #    Draw_hand_mode(t_xy, X, Y, S(t_xy))
                #else:
                ar_s = [search(s_interp, t_s, h) for h in t_xy]
                #ar_z = [search(ro_interp, w, h) for h in t_xy]
                Draw_hand_mode(t_xy, np.array(X), Y, np.array(ar_s), ro_points, w, t_z, z_points)
                if np.min(Y) < 0:
                    print("******************************************************************")
                    print("В процессе вычисления Y начал принимать недопустимые значения")
                C_1, C_2 = functionFBeta(X, s_points, X0, ro_points, w, t_xy, Y, T)
                print("******************************************************************")
                print("Качество", C_1+10*C_2)
                print("******************************************************************")
        else:
            array_xy = []
            qual = []
            for i in np.linspace(X0[0], X0[1], 3):
                for j in np.linspace(Y0[0], Y0[1], 3):
            #zp = list(zip(X0, Y0))
            #for pair in zp[:]:
                    B = BetaSearch(Bt, func1, func2, i, j, diff_z, U_y_interp, t_s, t_z, y_points, s_interp, 
                               s_points, ro_points, w, T)
                    if B != None:
                        array_xy.append([i, j, B[0], B[1], B[2], B[3], B[4], B[5]])
                        qual.append(B[4] + 10*B[5])
            print(qual)
            if len(qual) != 0:
                ind = np.argmin(qual)
                x0, y0, X, Y, t_xy, beta, c1, c2 = array_xy[ind]
                write_to_file(t_xy, X, "/home/asya/hell/x.txt")
                write_to_file(t_xy, Y, "/home/asya/hell/y.txt")
                print("******************************************************************")
                print("Качество", c1+10*c2)
                print("Оптимальное Bt", beta)
                print("Оптимальные X0, Y0", x0, y0)
                print("******************************************************************")
                if S != None:
                    Draw_hand_mode(t_xy, X, Y, S(t_xy), ro_points, w, t_z, z_points)
                else:
                    ar_s = [search(s_interp, t_s, h) for h in t_xy]
                    Draw_hand_mode(t_xy, np.array(X), Y, np.array(ar_s), ro_points, w, t_z, z_points)
                a = []
                arg = []
                Bt_ar = np.linspace(Bt[0], Bt[1], 10)
                for b in Bt_ar:
                    X, Y, t_xy = DiffEq(func1, func2, x0, y0, diff_z, U_y_interp, t_s, t_z, y_points, s_interp, b, T)
                    if np.min(Y) >= 0:
                        C_1, C_2 = functionFBeta(X, s_points, x0, ro_points, w, t_xy, Y, T)
                        a.append(C_1 + 10*C_2)
                        arg.append(b)
                if len(arg) != 0:
                    print(arg)
                    draw_F(arg, a)
            else:
                print("******************************************************************")
                print("В процессе вычисления произошло переполнение")
            #X, Y = DiffEq(diff_z, U_y_interp, f, B, X0, Y0[0], 0, 10)
            #C_1, C_2 = functionFBeta(X, s_interp, X0, ro_interp)
            #Draw_auto_mode(ro_interp, X, Y, s_interp, z_interp, C_1, C_2)
    #except Exception:
        #root.withdraw()
        #messagebox.showerror("Error", "Error message")
        #root.destroy()
        #.showinfo("Error", "Error")




# In[135]:

global Bt
global ro
#ro = 0
global S
#S = None
global f
global z
#z = None
global X0
global Y0
global T
w = []
t_s = []
t_z = []
ro_points = []
s_points = []
z_points = []
root = tkinter.Tk()

result = messagebox.askyesno("choose mode","Manual mode?")
Label(root, text="function ro(w): a*w*(b-w)").grid(row=0, column=0, columnspan=2, sticky=W+E+N+S, padx=5, pady=5)
Label(root, text="a").grid(row=1, column=0)
Label(root, text="b").grid(row=1, column=1)
e1 = Entry(root)
e2 = Entry(root)
e1.grid(row=2, column=0)
e2.grid(row=2, column=1)
bt_ro = Button(root, text="OK", command=callback_ro, height=1, width=10)
bt_ro.grid(row=2, column=2)

tx= Text(font=('times',12), wrap=WORD, width=15, height=2)
bt = Button(tx,text='Choose file', font=('times',12), width=15, height=2, command=get_filename_ro)
tx.grid(row=0, column=3)
tx.window_create(END,window=bt)
    
Label(root, text="function S(t): a*t + b*sin(t)").grid(row=4, column=0, columnspan=2, sticky=W+E+N+S, padx=5, pady=5)
Label(root, text="a").grid(row=5, column=0)
Label(root, text="b").grid(row=5, column=1)
e11 = Entry(root)
e22 = Entry(root)
e11.grid(row=6, column=0)
e22.grid(row=6, column=1)
bt_s = Button(root, text="OK", command=callback_S, height=1, width=10)
bt_s.grid(row=6, column=2)

tx2= Text(font=('times',12), wrap=WORD, width=15, height=2)
bt2 = Button(tx2,text='Choose file', font=('times',12), width=15, height=2, command=get_filename_s)
tx2.grid(row=4, column=3)
tx2.window_create(END,window=bt2)

Label(root, text="function z(t): a*t + b*cos(t)").grid(row=7, column=0, columnspan=2, sticky=W+E+N+S, padx=5, pady=5)
Label(root, text="a").grid(row=8, column=0)
Label(root, text="b").grid(row=8, column=1)
z_e1 = Entry(root)
z_e2 = Entry(root)
z_e1.grid(row=9, column=0)
z_e2.grid(row=9, column=1)
bt_z = Button(root, text="OK", command=callback_z, height=1, width=10)
bt_z.grid(row=9, column=2)

tx3= Text(font=('times',12), wrap=WORD, width=15, height=2)
bt3 = Button(tx3, text='Choose file', font=('times',12), width=15, height=2, command=get_filename_z)
tx3.grid(row=7, column=3)
tx3.window_create(END, window=bt3)



Label(root, text="Choose a family of functions for f").grid(row=10, column=0, columnspan=2, sticky=W+E+N+S, padx=5, pady=5)
var = IntVar()
R1 = Radiobutton(root, text="Constant", variable=var, value=1, command=sel)
R1.grid(row=11, column=0)
R2 = Radiobutton(root, text="Linear", variable=var, value=2, command=sel)
R2.grid(row=11, column=1)
label_sel = Label(root)
label_sel.grid(row=11, column=2)




if result:
    root.wm_title("Manual mode")
    Label(root, text="Value of Bt").grid(row=12, column=0, columnspan=2, sticky=W+E+N+S, padx=5, pady=5)
    bt_en = Entry(root)
    bt_en.grid(row=13, column=0)
    b2 = Button(root, text="OK", command=callback_Bt, height=1, width=10)
    b2.grid(row=13, column=2)

    Label(root, text="X0").grid(row=14, column=0)
    Label(root, text="Y0").grid(row=14, column=1)
    x_e1 = Entry(root)
    x_e2 = Entry(root)
    x_e1.grid(row=15, column=0)
    x_e2.grid(row=15, column=1)
    bt_x = Button(root, text="OK", command=callback_x, height=1, width=10)
    bt_x.grid(row=15, column=2)
    
else:
    root.wm_title("Auto mode")
    Label(root, text="Interval for Bt").grid(row=12, column=0, columnspan=2, sticky=W+E+N+S, padx=5, pady=5)
    bt_en1 = Entry(root)
    bt_en1.grid(row=13, column=0)
    bt_en2 = Entry(root)
    bt_en2.grid(row=13, column=1)
    b2 = Button(root, text="OK", command=callback_Bt_int, height=1, width=10)
    b2.grid(row=13, column=2)
    Label(root, text="X0_1").grid(row=14, column=0)
    Label(root, text="X0_2").grid(row=14, column=1)
    Label(root, text="Y0_1").grid(row=14, column=3)
    Label(root, text="Y0_2").grid(row=14, column=4)
    x_e1 = Entry(root)
    x_e2 = Entry(root)
    x_e1.grid(row=15, column=0)
    x_e2.grid(row=15, column=1)
    bt_x = Button(root, text="OK", command=callback_x2, height=1, width=10)
    bt_x.grid(row=15, column=2)
    
    xy1 = Entry(root)
    xy2 = Entry(root)
    xy1.grid(row=15, column=3)
    xy2.grid(row=15, column=4)
    bt_xy = Button(root, text="OK", command=callback_y2, height=1, width=10)
    bt_xy.grid(row=15, column=5)


Label(root, text="T").grid(row=16, column=0, columnspan=2, sticky=W+E+N+S, padx=5, pady=5)
bt_T1 = Entry(root)
bt_T1.grid(row=17, column=0)
bt_T2 = Button(root, text="OK", command=callback_Bt_T, height=1, width=10)
bt_T2.grid(row=17, column=2)

Button(root, text="Solver", command=Solver, height=1, width=20).grid(row=18, column=0, columnspan=2, sticky=W+E+N+S)    

root.mainloop()
