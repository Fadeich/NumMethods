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


from matplotlib.figure import Figure
root = tkinter.Tk()




def get_points(array):
    root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print (root.filename)
    if len(root.filename):
        try:
            file = open(root.filename)
            for line in file:
                array.append((int(line.strip()[0]), int(line.strip()[1])))
        except Exception:
            root.withdraw()
            messagebox.showerror("Error", "Error message")


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
    except Exception:
        root.withdraw()
        messagebox.showerror("Error", "Error message")
    global ro 
    ro = (lambda w: a*w*(b-w))
    bt_ro['text'] = "Saved"

def callback_S():
    try:
       a = float(e11.get())
       b = float(e22.get())
    except Exception:
        root.withdraw()
        messagebox.showerror("Error", "Error message")
    global S 
    S = (lambda t: a*t + b*np.sin(t))
    bt_s['text'] = "Saved"

def callback_z():
    try:
       a = float(z_e1.get())
       b = float(z_e2.get())
    except Exception:
        root.withdraw()
        messagebox.showerror("Error", "Error message")
    global z 
    z = (lambda t: a*t + b*np.cos(t))
    bt_z['text'] = "Saved"

def callback_Bt():
    global Bt
    try:
        Bt = float(bt_en.get())
        print(Bt)
    except Exception:
        root.withdraw()
        messagebox.showerror("Error", "Error message")
    b2['text'] = "Saved"

def callback_Bt_int():
    global Bt
    try:
        Bt = [min(float(bt_en1.get()), float(bt_en2.get())), max(float(bt_en1.get()), float(bt_en2.get()))]
        print(Bt)
    except Exception:
        root.withdraw()
        messagebox.showerror("Error", "Error message")
    b2['text'] = "Saved"

def callback_x():
    global X0
    global Y0
    try:
        X0 = float(x_e1.get())
        Y0 = float(x_e2.get())
        print("X0:", X0, "Y0:", Y0)
    except Exception:
        root.withdraw()
        messagebox.showerror("Error", "Error message")
    bt_x['text'] = "Saved"

def callback_x2():
    global X0
    global Y0
    try:
        X0 = S(0)
        Y0 = [float(x_e1.get()), float(x_e2.get())]
        print(X0, Y0)
    except Exception:
        root.withdraw()
        messagebox.showerror("Error", "Error message")
    bt_x['text'] = "Saved"

def sel():
    selection = "You selected the option " + str(var.get())
    label_sel.config(text = selection)
    global f
    if var.get() == 1:
        f = (lambda z, x, S, Bt: Bt)
    else:
        f = (lambda z, x, S, Bt: Bt*(x-z))

def Tabulate(f):
    print("Tabulate")
    array = []
    for i in range(1000):
        array.append((i, f(i)))
    return array

def Integrate(f, a, b):
    print("Integrate")
    return (lambda t: 3*t**2 - 2*t**3 -1)

def TabulateIntegral(f):
    print("TabulateIntegral")
    y = 0
    U_y = Integrate(f, 1, y)
    return Tabulate(U_y)

def LinSys(A, b):
    print("LinSys")
    return np.random.randint(4)

def Interp(points):
    print("Interp")
    A = np.random.randint(10, size=(4, 4))
    b = np.random.randint(4)
    x = LinSys(A, b)
    return (lambda t: 3*t**2)

def Diff(f):
    print("Diff")
    return (lambda t: 6*t)

def DiffEq(diff_z, U_y_interp, f, Bt, X0, Y0):
    print("DiffEq")
    A = np.random.randint(10, size=(4, 4))
    b = np.random.randint(4)
    x = LinSys(A, b)
    array = []
    return array, array

def functionFBeta(X, s_interp, X_0, ro_interp):
    print("functionFBeta")
    Integrate(ro_interp, 1, 2)
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

def Draw_hand_mode(X, Y, S):
    wdw1 = Toplevel()
    wdw1.geometry('700x700')
    f = Figure(figsize=(10, 10), dpi=100)
    a = f.add_subplot(111)
    t = arange(1000)
    c = arange(1000)
    a.plot(t, c**3, 'r', label = r'$x(t)$')
    a0 = f.add_subplot(111)
    a0.plot(t, t**2, 'b', label = r'$S(t)$')
    a.set_title('dynamics')
    a.set_xlabel('t')
    a.set_ylabel('X(t), S(t)')
    a.legend(loc = 0, fontsize = 18)
    canvas = FigureCanvasTkAgg(f, master=wdw1)
    canvas.show()
    canvas.get_tk_widget().pack()

    wdw2 = Toplevel()
    wdw2.geometry('700x700')
    f2 = Figure(figsize=(10, 10), dpi=100)
    a2 = f2.add_subplot(111)
    a2.plot(t, t**2, 'b', label = r'$y(t)$')
    a2.set_title('dynamics')
    a2.set_xlabel('t')
    a2.set_ylabel('Y(y)')
    a2.legend(loc = 0, fontsize = 18)
    canvas2 = FigureCanvasTkAgg(f2, master=wdw2)
    canvas2.show()
    canvas2.get_tk_widget().pack()

def Draw_auto_mode(X, Y, S):
    wdw1 = Toplevel()
    wdw1.geometry('700x700')
    f = Figure(figsize=(10, 10), dpi=100)
    a = f.add_axes([0.1, 0.1, 0.8, 0.8])
    t = arange(1000)
    a.plot(t, t**2, 'b', label = r'$10*C_1+C_2$')
    a.set_title('dynamics')
    a.set_xlabel('Beta')
    a.set_ylabel('F=10*C_1+C_2')
    a.legend(loc = 0, fontsize = 18)
    canvas = FigureCanvasTkAgg(f, master=wdw1)
    canvas.show()
    canvas.get_tk_widget().pack()

    wdw2 = Toplevel()
    wdw2.geometry('700x700')
    f2 = Figure(figsize=(10, 10), dpi=100)
    a2 = f2.add_subplot(111)
    a2.plot(t, t**2, 'b', label = r'$y(t)$')
    a2.set_title('dynamics')
    a2.set_xlabel('t')
    a2.set_ylabel('Y(y)')
    a2.legend(loc = 0, fontsize = 18)
    canvas2 = FigureCanvasTkAgg(f2, master=wdw2)
    canvas2.show()
    canvas2.get_tk_widget().pack()

    wdw3 = Toplevel()
    wdw3.geometry('700x700')
    f3 = Figure(figsize=(10, 10), dpi=100)
    a3 = f3.add_subplot(111)
    a3.plot(t, t**2, 'b', label = r'$S(t) - x(t)$')
    a3.set_title('dynamics')
    a3.set_xlabel('t')
    a3.set_ylabel('S(t) - x(t)')
    a3.legend(loc = 0, fontsize = 18)
    canvas3 = FigureCanvasTkAgg(f3, master=wdw3)
    canvas3.show()
    canvas3.get_tk_widget().pack()


def Solver():
    print("Solver")
    ro_points = Tabulate(ro)
    s_points = Tabulate(S)
    z_points = Tabulate(z)
    U_y_points = TabulateIntegral(ro)
    ro_interp = Interp(ro_points)
    s_interp = Interp(s_points)
    z_interp = Interp(z_points)
    U_y_interp = Interp(U_y_points)
    diff_z = Diff(z_interp)
    if result:
        X, Y = DiffEq(diff_z, U_y_interp, f, Bt, X0, Y0)
        C_1, C_2 = functionFBeta(X, s_interp, X0, ro_interp)
        Draw_hand_mode(X, Y, s_interp)
    else:
        #global X0
        B = BetaSearch(Bt, diff_z, U_y_interp, f, X0, Y0[0], s_interp, ro_interp)
        X, Y = DiffEq(diff_z, U_y_interp, f, B, X0, Y0[0])
        Draw_auto_mode(X, Y, s_interp)

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
