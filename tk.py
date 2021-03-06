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
root = tkinter.Tk()




def get_points(array):
    root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print (root.filename)
    if len(root.filename):
        try:
            f = open(root.filename)
            for line in f:
                print(line)
                array.append((int(line.split()[0]), int(line.split()[1])))
        except Exception:
            #root.withdraw()
            #messagebox.showerror("Error", "Error message")
            #root.destroy()
            messagebox.showinfo("Error", "Error")


def get_filename_ro():
    get_points(ro_points)

def get_filename_s():
    get_points(s_points)

def get_filename_z():
    get_points(z_points)

def callback_ro():
    try:
        a = float(e1.get())
        b = float(e2.get())
        global ro 
        ro = (lambda w: a*w*(b-w))
        bt_ro['text'] = "Saved"
    except Exception:
        #root.withdraw()
        #messagebox.showerror("Error", "Error message")
        #root.destroy()
        messagebox.showinfo("Error", "Error")
    

def callback_S():
    try:
       a = float(e11.get())
       b = float(e22.get())
       global S 
       S = (lambda t: a*t + b*np.sin(t))
       bt_s['text'] = "Saved"
    except Exception:
        #root.withdraw()
        #messagebox.showerror("Error", "Error message")
        #root.destroy
        messagebox.showinfo("Error", "Error")

def callback_z():
    try:
       a = float(z_e1.get())
       b = float(z_e2.get())
       global z
       z = (lambda t: a*t + b*np.cos(t))
       bt_z['text'] = "Saved"
    except Exception:
        #root.withdraw()
        #messagebox.showerror("Error", "Error message")
        #root.destroy()
        messagebox.showinfo("Error", "Error")
    
    

def callback_Bt():
    global Bt
    try:
        Bt = float(bt_en.get())
        print(Bt)
        b2['text'] = "Saved"
    except Exception:
        #root.withdraw()
        #messagebox.showerror("Error", "Error message")
        #root.destroy()
        messagebox.showinfo("Error", "Error")
    

def callback_Bt_int():
    global Bt
    try:
        Bt = [min(float(bt_en1.get()), float(bt_en2.get())), max(float(bt_en1.get()), float(bt_en2.get()))]
        print(Bt)
        b2['text'] = "Saved"
    except Exception:
        #root.withdraw()
        #messagebox.showerror("Error", "Error message")
        #root.destroy()
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
        #root.withdraw()
        #messagebox.showerror("Error", "Error message")
        #root.destroy()
        messagebox.showinfo("Error", "Error")
    

def callback_x2():
    global X0
    global Y0
    try:
        X0 = S(0)
        Y0 = [float(x_e1.get()), float(x_e2.get())]
        print(X0, Y0)
        bt_x['text'] = "Saved"
    except Exception:
        #root.withdraw()
        #messagebox.showerror("Error", "Error message")
        #root.destroy()
        messagebox.showinfo("Error", "Error")
    

def sel():
    selection = "You selected the option " + str(var.get())
    label_sel.config(text = selection)
    global f
    if var.get() == 1:
        f = (lambda z, x, S, Bt: Bt)
    else:
        f = (lambda z, x, S, Bt: Bt*(x-z))

def write_to_file(array, path):
    f = open(path, 'w')
    for pair in array:
        f.write(str(pair[0])+" "+ str(pair[1]) + '\n')

def write_to_file2(array, path):
    f = open(path, 'w')
    for a in array:
        f.write(str(a)+" ")

def Tabulate(f, path = ""):
    print("Tabulate")
    array = []
    for i in range(1000):
        array.append((i, f(i)))
    if len(path):
        write_to_file(array, path)
    return array

def Integrate(x, f, a, b):
    print("Integrate")
    x = x[a <= x]
    print("hey")
    x = x[x <= b]
    print("hey2")
    sum_int = f[0]+f[-1]+2*sum(f[1:-1])
    print("hey3")
    return float(sum_int*(x[1]-x[0])/2)

#def Integrate(f, a, b):
#    print("Integrate")
#    return (lambda t: 3*t**2 - 2*t**3 -1)

def TabulateIntegral(f, path = ""):
    print("TabulateIntegral")
    #x = np.linspace(0, 1, 10)
    #y = np.array([i**2 for i in x])
    ro_w_x = np.linspace(1, 2, 10)
    ro_w_y = np.array([i**2 for i in ro_w_x])
    U_y = Integrate(ro_w_x, ro_w_y, 1, 2)
    #U_y = []
    #for y in range(1, 10):
    #    U_y.append(Integrate(x, f, 0, y))
    return U_y

def LinSys(A, b):
    print("LinSys")
    return np.random.randint(4)

def Interp(points, path = ""):
    print("Interp")
    A = np.random.randint(10, size=(4, 4))
    b = np.random.randint(4)
    x = LinSys(A, b)
    
    return (lambda t: 3*t**2)

def Diff(f):
    print("Diff")
    return (lambda t: 6*t)

def DiffEq(diff_z, U_y_interp, f, Bt, X0, Y0, t1, t2):
    print("DiffEq")
    A = np.random.randint(10, size=(4, 4))
    b = np.random.randint(4)
    x = LinSys(A, b)
    array_x = []
    array_y = []
    f_prev1 = 0
    f_prev2 = 0
    x_prev = X0
    y_prev = Y0
    for i in range(t1, t2):
        x_prev = f_prev1 + x_prev
        array_x.append(x_prev)
        y_prev = f_prev2 + y_prev
        array_y.append(y_prev)
    print(array_x, array_y)
    write_to_file2(array_x, "/home/asya/hell/x.txt")
    write_to_file2(array_y, "/home/asya/hell/y.txt")
    return array_x, array_y

def functionFBeta(X, s_interp, X_0, ro_interp):
    print("functionFBeta")
    ro_w_x = np.linspace(1, 2, 10)
    ro_w_y = np.array([i**2 for i in ro_w_x])
    U_y = Integrate(ro_w_x, ro_w_y, 1, 2)
    #Integrate(ro_interp, 1, 2)
   
    return 0.9, 0.5

def BetaSearch(Bt, diff_z, U_y_interp, f, X0, Y0_1, s_interp, ro_interp):
        ar = arange(Bt[0], Bt[1], 0.1)
        X_prev, Y_prev = DiffEq(diff_z, U_y_interp, f, Bt, X0, Y0)
        C_1_prev, C_2_prev = functionFBeta(X_prev, s_interp, X0, ro_interp)
        B_min = Bt[0]
        for B in ar:
            X, Y = DiffEq(diff_z, U_y_interp, f, B, X0, Y0)
            C_1, C_2 = functionFBeta(X, s_interp, X0, ro_interp)
            if 10*C_1 + C_2 < 10*C_1_prev + C_2_prev:
                X, Y, C_1, C_2, B_min = X_prev, Y_prev, C_1_prev, C_2_prev, B
        return B_min

def draw_ro(Ro):
    wdw1 = Toplevel()
    wdw1.geometry('700x700')
    f = Figure(figsize=(10, 10), dpi=100)
    a = f.add_axes([0.1, 0.1, 0.8, 0.8])
    t = arange(1000)
    a.plot(t, t**2, 'b', label = r'$ro(w)$')
    a.set_title('dynamics')
    a.set_xlabel('w')
    a.set_ylabel('ro')
    a.legend(loc = 0, fontsize = 18)
    canvas = FigureCanvasTkAgg(f, master=wdw1)
    canvas.show()
    canvas.get_tk_widget().pack()

def draw_x_S(X, S):
    wdw2 = Toplevel()
    wdw2.geometry('700x700')
    f2 = Figure(figsize=(10, 10), dpi=100)
    a = f2.add_axes([0.1, 0.1, 0.8, 0.8])
    t = arange(1000)
    a.plot(t, t**2, 'b', label = r'$x(t)$')
    a.plot(t, t, 'r', label = r'$S(t)$')
    a.set_title('dynamics')
    a.set_xlabel('t')
    a.set_ylabel('S(t), x(t)')
    a.legend(loc = 0, fontsize = 18)
    canvas = FigureCanvasTkAgg(f2, master=wdw2)
    canvas.show()
    canvas.get_tk_widget().pack()

def draw_y(Y):
    wdw3 = Toplevel()
    wdw3.geometry('700x700')
    f3 = Figure(figsize=(10, 10), dpi=100)
    a = f3.add_axes([0.1, 0.1, 0.8, 0.8])
    t = arange(1000)
    a.plot(t, t**2, 'b', label = r'$y(t)$')
    a.set_title('dynamics')
    a.set_xlabel('t')
    a.set_ylabel('y(t)')
    a.legend(loc = 0, fontsize = 18)
    canvas = FigureCanvasTkAgg(f3, master=wdw3)
    canvas.show()
    canvas.get_tk_widget().pack()

def draw_z(Y):
    wdw4 = Toplevel()
    wdw4.geometry('700x700')
    f4 = Figure(figsize=(10, 10), dpi=100)
    a = f4.add_axes([0.1, 0.1, 0.8, 0.8])
    t = arange(1000)
    a.plot(t, t**2, 'b', label = r'$z(t)$')
    a.set_title('dynamics')
    a.set_xlabel('t')
    a.set_ylabel('z(t)')
    a.legend(loc = 0, fontsize = 18)
    canvas = FigureCanvasTkAgg(f4, master=wdw4)
    canvas.show()
    canvas.get_tk_widget().pack()

def draw_xS(X, S):
    wdw5 = Toplevel()
    wdw5.geometry('700x700')
    f5 = Figure(figsize=(10, 10), dpi=100)
    a = f5.add_axes([0.1, 0.1, 0.8, 0.8])
    t = arange(1000)
    a.plot(t, t**2, 'b', label = r'$S-x$')
    a.set_title('dynamics')
    a.set_xlabel('t')
    a.set_ylabel('S(t), x(t)')
    a.legend(loc = 0, fontsize = 18)
    canvas = FigureCanvasTkAgg(f5, master=wdw5)
    canvas.show()
    canvas.get_tk_widget().pack()

def draw_Sx(X, S):
    wdw6 = Toplevel()
    wdw6.geometry('700x700')
    f6 = Figure(figsize=(10, 10), dpi=100)
    a = f6.add_axes([0.1, 0.1, 0.8, 0.8])
    t = arange(1000)
    a.plot(t, t**2, 'b', label = r'$S(x)$')
    a.set_title('dynamics')
    a.set_xlabel('x')
    a.set_ylabel('S(x)')
    a.legend(loc = 0, fontsize = 18)
    canvas = FigureCanvasTkAgg(f6, master=wdw6)
    canvas.show()
    canvas.get_tk_widget().pack()

def draw_F(C_1, C_2):
    wdw7 = Toplevel()
    wdw7.geometry('700x700')
    f7 = Figure(figsize=(10, 10), dpi=100)
    a = f7.add_axes([0.1, 0.1, 0.8, 0.8])
    t = arange(1000)
    a.plot(t, t**2, 'b', label = r'$C_2$')
    a.plot(t, t, 'y', label = r'$C_1$')
    a.set_title('dynamics')
    a.set_xlabel('x')
    a.set_ylabel('S(x)')
    a.legend(loc = 0, fontsize = 18)
    canvas = FigureCanvasTkAgg(f7, master=wdw7)
    canvas.show()
    canvas.get_tk_widget().pack()

def Draw_hand_mode(Ro, X, Y, S, Z):
    draw_ro(Ro)
    draw_x_S(X, S)
    draw_y(Y)
    draw_z(Z)
    draw_xS(X, S)
    draw_Sx(X, S)


def Draw_auto_mode(Ro, X, Y, S, Z, C_1, C_2):
    draw_ro(Ro)
    draw_x_S(X, S)
    draw_y(Y)
    draw_z(Z)
    draw_xS(X, S)
    draw_Sx(X, S)
    draw_F(C_1, C_2)


def Solver():
    try:
        print(Bt)
        print(ro)
        print(S)
        print(f)
        print(z)
        print(X0, Y0)
        print("Solver")
        ro_points = Tabulate(ro, "/home/asya/hell/ro.txt")
        s_points = Tabulate(S, "/home/asya/hell/S.txt")
        z_points = Tabulate(z, "/home/asya/hell/z.txt")
        U_y_points = TabulateIntegral(ro, "/home/asya/hell/U_y.txt")
        ro_interp = Interp(ro_points)
        s_interp = Interp(s_points)
        z_interp = Interp(z_points)
        U_y_interp = Interp(U_y_points)
        diff_z = Diff(z_interp)
        if result:
            X, Y = DiffEq(diff_z, U_y_interp, f, Bt, X0, Y0, 0, 10) #t1 t2
            C_1, C_2 = functionFBeta(X, s_interp, X0, ro_interp)
            Draw_hand_mode(ro_interp, X, Y, s_interp, z_interp)
        else:
            B = BetaSearch(Bt, diff_z, U_y_interp, f, X0, Y0[0], s_interp, ro_interp)
            X, Y = DiffEq(diff_z, U_y_interp, f, B, X0, Y0[0], 0, 10)
            C_1, C_2 = functionFBeta(X, s_interp, X0, ro_interp)
            Draw_auto_mode(ro_interp, X, Y, s_interp, z_interp, C_1, C_2)
    except Exception:
        #root.withdraw()
        #messagebox.showerror("Error", "Error message")
        #root.destroy()
        messagebox.showinfo("Error", "Error")

global Bt
global ro
global S
global f
global z
global X0
global Y0
ro_points = []
s_points = []
z_points = []

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


Label(root, text="Choose a family of functions for f").grid(row=7, column=0, columnspan=2, sticky=W+E+N+S, padx=5, pady=5)
var = IntVar()
R1 = Radiobutton(root, text="Constant", variable=var, value=1, command=sel)
R1.grid(row=8, column=0)
R2 = Radiobutton(root, text="Linear", variable=var, value=2, command=sel)
R2.grid(row=8, column=1)
label_sel = Label(root)
label_sel.grid(row=8, column=2)
if result:
    root.wm_title("Manual mode")
    Label(root, text="Value of Bt").grid(row=9, column=0, columnspan=2, sticky=W+E+N+S, padx=5, pady=5)
    bt_en = Entry(root)
    bt_en.grid(row=10, column=0)
    b2 = Button(root, text="OK", command=callback_Bt, height=1, width=10)
    b2.grid(row=10, column=2)
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
    Label(root, text="Interval for Bt").grid(row=9, column=0, columnspan=2, sticky=W+E+N+S, padx=5, pady=5)
    bt_en1 = Entry(root)
    bt_en1.grid(row=10, column=0)
    bt_en2 = Entry(root)
    bt_en2.grid(row=10, column=1)
    b2 = Button(root, text="OK", command=callback_Bt_int, height=1, width=10)
    b2.grid(row=10, column=2)
    Label(root, text="Y0_1").grid(row=14, column=0)
    Label(root, text="Y0_2").grid(row=14, column=1)
    x_e1 = Entry(root)
    x_e2 = Entry(root)
    x_e1.grid(row=15, column=0)
    x_e2.grid(row=15, column=1)
    bt_x = Button(root, text="OK", command=callback_x2, height=1, width=10)
    bt_x.grid(row=15, column=2)

Label(root, text="function z(t): a*t + b*cos(t)").grid(row=11, column=0, columnspan=2, sticky=W+E+N+S, padx=5, pady=5)
Label(root, text="a").grid(row=12, column=0)
Label(root, text="b").grid(row=12, column=1)
z_e1 = Entry(root)
z_e2 = Entry(root)
z_e1.grid(row=13, column=0)
z_e2.grid(row=13, column=1)
bt_z = Button(root, text="OK", command=callback_z, height=1, width=10)
bt_z.grid(row=13, column=2)

    #tx3= Text(font=('times',12), wrap=WORD, width=15, height=2)
    #bt3 = Button(tx2,text='Choose file', font=('times',12), width=15, height=2, command=get_filename_z)
    #tx3.grid(row=11, column=3)
    #tx3.window_create(END, window=bt3)
    
Button(root, text="Solver", command=Solver, height=1, width=20).grid(row=16, column=0, columnspan=2, sticky=W+E+N+S)

    
    
    

root.mainloop()
