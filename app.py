#    source /home/aymen/Desktop/ABSAO/.env/bin/activate

import streamlit as st
import datetime as dt
import pandas as pd
import numpy as np
import base64
from pathlib import Path
import time
import base64
from itertools import chain
from pyomo.environ import *

import pyutilib.subprocess.GlobalData
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

#image Al Boraq
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
    img_to_bytes("./image/bg.png")
)
st.markdown(
    header_html, unsafe_allow_html=True,
)



@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('./image/test.png')


main_bg = "./image/test.png"
main_bg_ext = "png"

side_bg = "./image/test.png"
side_bg_ext = "png"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)


#Limite de dates
minDate = dt.date(2022, 1, 1)
maxDate = dt.date(2022, 3, 31)

#insertion de la date
date1= st.date_input('Entrez la date du voyage',
                     minDate , 
                     min_value=minDate, 
                     max_value=maxDate,)

#insertion du tempd
options=np.array(["6:00","7:00","8:00","9:00","10:00","11:00","12:00",
                "13:00","14:00","15:00","16:00","17:00","18:00",
                "19:00","20:00","21:00"])

heure =st.selectbox(
    "Entrez l'heure du voyage"
    ,options)

#insertion de la direction
Dir = st.radio(
     "Selectionnez le direction du train",
     ('CASA VOYAGEURS ---> TANGER VILLE ', 'TANGER VILLE ---> CASA VOYAGEURS'))




if st.button('Afficher les allocations optimales'):        
    with st.spinner('Please wait'):
        st.write("Forecast loading ....")
        delta = date1 - minDate
        z=delta.days
        obj = [115,115,115,82,80,80,80,58,90,90,90,63,50,50,50,38,364,299,243,210,224,189,149,132,244,199,162,129,139,116,93,89,26,26,26,20,18,18,18,14,281,234,187,129,172,143,115,89,115,115,115,82,80,80,80,58,90,90,90,63,50,50,50,38,364,299,243,210,224,189,149,132,244,199,162,129,139,116,93,89,26,26,26,20,18,18,18,14,281,234,187,129,172,143,115,89]
        obj=np.array(obj)
        s=50

        if heure=="6:00":
            with open('./Data/6.npy', 'rb') as f:
                a = np.load(f, allow_pickle=True)
            dd=a.tolist()
            a=[0]*90
            for k in range(32,40):
                dd[k]=a
            for k in range(80,88):
                dd[k]=a
            d=[]
            for i in range(96):
                d.append(dd[i][z])

        if heure=="7:00":
            with open('./Data/7.npy', 'rb') as f:
                a = np.load(f, allow_pickle=True)
            dd=a.tolist()
            a=[0]*90
            for k in range(32,40):
                dd[k]=a
            for k in range(80,88):
                dd[k]=a            
            d=[]
            for i in range(96):
                d.append(dd[i][z])

        if heure=="8:00":
            with open('./Data/8.npy', 'rb') as f:
               a = np.load(f, allow_pickle=True)
            dd=a.tolist()
            a=[0]*90
            for k in range(32,40):
                dd[k]=a
            for k in range(80,88):
                dd[k]=a 
            d=[]
            for i in range(96):
                d.append(dd[i][z])

        if heure=="9:00":
            with open('./Data/9.npy', 'rb') as f:
                a = np.load(f, allow_pickle=True)
            dd=a.tolist()
            a=[0]*90
            for k in range(32,40):
                dd[k]=a
            for k in range(80,88):
                dd[k]=a
            d=[]
            for i in range(96):
                d.append(dd[i][z])

        if heure=="10:00":
            with open('./Data/10.npy', 'rb') as f:
                a = np.load(f, allow_pickle=True)
            dd=a.tolist()
            a=[0]*90
            for k in range(32,40):
                dd[k]=a
            for k in range(80,88):
                dd[k]=a
            d=[]
            for i in range(96):
                d.append(dd[i][z])

        if heure=="11:00":
            with open('./Data/11.npy', 'rb') as f:
                a = np.load(f, allow_pickle=True)
            dd=a.tolist()
            a=[0]*90
            for k in range(32,40):
                dd[k]=a
            for k in range(80,88):
                dd[k]=a
            d=[]
            for i in range(96):
                d.append(dd[i][z])

        if heure=="12:00":
            with open('./Data/12.npy', 'rb') as f:
                a = np.load(f, allow_pickle=True)
            dd=a.tolist()
            a=[0]*90
            for k in range(32,40):
                dd[k]=a
            for k in range(80,88):
                dd[k]=a
            d=[]
            for i in range(96):
                d.append(dd[i][z])

        if heure=="13:00":
            with open('./Data/13.npy', 'rb') as f:
                a = np.load(f, allow_pickle=True)
            dd=a.tolist()
            a=[0]*90
            for k in range(32,40):
                dd[k]=a
            for k in range(80,88):
                dd[k]=a
            d=[]
            for i in range(96):
                d.append(dd[i][z])

        if heure=="14:00":
            with open('./Data/14.npy', 'rb') as f:
                a = np.load(f, allow_pickle=True)
            dd=a.tolist()
            a=[0]*90
            for k in range(32,40):
                dd[k]=a
            for k in range(80,88):
                dd[k]=a
            d=[]
            for i in range(96):
                d.append(dd[i][z])

        if heure=="15:00":
            with open('./Data/15.npy', 'rb') as f:
                a = np.load(f, allow_pickle=True)
            dd=a.tolist()
            a=[0]*90
            for k in range(32,40):
                dd[k]=a
            for k in range(80,88):
                dd[k]=a
            d=[]
            for i in range(96):
                d.append(dd[i][z])

        if heure=="16:00":
            with open('./Data/16.npy', 'rb') as f:
                a = np.load(f, allow_pickle=True)
            dd=a.tolist()
            a=[0]*90
            for k in range(32,40):
                dd[k]=a
            for k in range(80,88):
                dd[k]=a
            d=[]
            for i in range(96):
                d.append(dd[i][z])

        if heure=="17:00":
            with open('./Data/17.npy', 'rb') as f:
                a = np.load(f, allow_pickle=True)
            dd=a.tolist()
            a=[0]*90
            for k in range(32,40):
                dd[k]=a
            for k in range(80,88):
                dd[k]=a
            d=[]
            for i in range(96):
                d.append(dd[i][z])

        if heure=="18:00":
            with open('./Data/18.npy', 'rb') as f:
                a = np.load(f, allow_pickle=True)
            dd=a.tolist()
            a=[0]*90
            for k in range(32,40):
                dd[k]=a
            for k in range(80,88):
                dd[k]=a 
            d=[]
            for i in range(96):
                d.append(dd[i][z])

        if heure=="19:00":
            with open('./Data/19.npy', 'rb') as f:
                a = np.load(f, allow_pickle=True)
            dd=a.tolist()
            d=[]
            for i in range(96):
                d.append(dd[i][z])

        if heure=="20:00":
            with open('./Data/20.npy', 'rb') as f:
                a = np.load(f, allow_pickle=True)
            dd=a.tolist()
            a=[0]*90
            for k in range(32,40):
                dd[k]=a
            for k in range(80,88):
                dd[k]=a
            d=[]
            for i in range(96):
                d.append(dd[i][z])

        if heure=="21:00":
            with open('./Data/21.npy', 'rb') as f:
                a = np.load(f, allow_pickle=True)
            dd=a.tolist()
            a=[0]*90
            for k in range(32,40):
                dd[k]=a
            for k in range(80,88):
                dd[k]=a
            d=[]
            for i in range(96):
                d.append(dd[i][z])

        st.write("Optimization loading ....")
        d=np.array(d)
        id = np.identity(96, dtype = int)
        #contrainte de roulement
        ones4=np.ones(4, dtype=int)
        zeros4=np.zeros(4, dtype=int)
        zeros24=np.zeros(24, dtype=int)
        zeros48=np.zeros(48, dtype=int)
        c11=np.concatenate((ones4, zeros4, ones4, zeros4 ,ones4, zeros4, zeros24, zeros48), axis=None)
        c21=np.concatenate((zeros4, ones4, zeros4, ones4, zeros4 ,ones4, zeros24, zeros48), axis=None)
        c31=np.concatenate((ones4, zeros4, zeros4, zeros4, ones4, zeros4, zeros4, zeros4, ones4, zeros4, ones4, zeros4 , zeros48), axis=None)
        c41=np.concatenate((zeros4, ones4, zeros4, zeros4, zeros4, ones4, zeros4, zeros4, zeros4, ones4, zeros4, ones4 , zeros48), axis=None)
        c51=np.concatenate((zeros4, zeros4, zeros4, zeros4 , ones4, zeros4, ones4, zeros4, zeros4, zeros4, ones4, zeros4, zeros48), axis=None)
        c61=np.concatenate((zeros4, zeros4, zeros4, zeros4 , zeros4, ones4, zeros4, ones4, zeros4, zeros4, zeros4, ones4, zeros48), axis=None)


        c12=np.concatenate((zeros48, ones4, zeros4, ones4, zeros4 ,ones4, zeros4, zeros24), axis=None)
        c22=np.concatenate((zeros48, zeros4, ones4, zeros4, ones4, zeros4 ,ones4, zeros24), axis=None)
        c32=np.concatenate((zeros48, ones4, zeros4, zeros4, zeros4, ones4, zeros4, zeros4, zeros4, ones4, zeros4, ones4, zeros4), axis=None)
        c42=np.concatenate((zeros48, zeros4, ones4, zeros4, zeros4, zeros4, ones4, zeros4, zeros4, zeros4, ones4, zeros4, ones4), axis=None)
        c52=np.concatenate((zeros48, zeros4, zeros4, zeros4, zeros4 , ones4, zeros4, ones4, zeros4, zeros4, zeros4, ones4, zeros4), axis=None)
        c62=np.concatenate((zeros48, zeros4, zeros4, zeros4, zeros4 , zeros4, ones4, zeros4, ones4, zeros4, zeros4, zeros4, ones4), axis=None)
        #contrainte de priorite
        ct11 = np.concatenate((np.zeros(16, dtype=int), np.ones( 4, dtype=int), np.zeros(28, dtype=int), np.zeros(48, dtype=int)), axis=None)
        ct21 = np.concatenate((np.zeros(20, dtype=int), np.ones( 4, dtype=int), np.zeros(24, dtype=int), np.zeros(48, dtype=int)), axis=None)
        rt11 = np.concatenate((np.zeros(40, dtype=int), np.ones( 4, dtype=int), np.zeros( 4, dtype=int), np.zeros(48, dtype=int)), axis=None)
        rt21 = np.concatenate((np.zeros(44, dtype=int), np.ones( 4, dtype=int), np.zeros(48, dtype=int)), axis=None)
        kt11 = np.concatenate((np.zeros(24, dtype=int), np.ones( 4, dtype=int), np.zeros(20, dtype=int), np.zeros(48, dtype=int)), axis=None)
        kt21 = np.concatenate((np.zeros(28, dtype=int), np.ones( 4, dtype=int), np.zeros(16, dtype=int), np.zeros(48, dtype=int)), axis=None)
        ck11 = np.concatenate((np.ones( 4, dtype=int), np.zeros(44, dtype=int), np.zeros(48, dtype=int)), axis=None)
        ck21 = np.concatenate((np.zeros( 4, dtype=int), np.ones( 4, dtype=int), np.zeros(40, dtype=int), np.zeros(48, dtype=int)), axis=None)
        cr11 = np.concatenate((np.zeros( 8, dtype=int), np.ones( 4, dtype=int), np.zeros(36, dtype=int), np.zeros(48, dtype=int)), axis=None)
        cr21 = np.concatenate((np.zeros(12, dtype=int), np.ones( 4, dtype=int), np.zeros(32, dtype=int), np.zeros(48, dtype=int)), axis=None)
        rk11 = np.concatenate((np.zeros(32, dtype=int), np.ones( 4, dtype=int), np.zeros(12, dtype=int), np.zeros(48, dtype=int)), axis=None)
        rk21 = np.concatenate((np.zeros(36, dtype=int), np.ones( 4, dtype=int), np.zeros( 8, dtype=int), np.zeros(48, dtype=int)), axis=None)

        ct12 = np.concatenate((np.zeros(16, dtype=int), np.ones( 4, dtype=int), np.zeros(28, dtype=int), np.zeros(48, dtype=int)), axis=None)
        ct22 = np.concatenate((np.zeros(48, dtype=int), np.zeros(20, dtype=int), np.ones( 4, dtype=int), np.zeros(24, dtype=int)), axis=None)
        rt12 = np.concatenate((np.zeros(48, dtype=int), np.zeros(40, dtype=int), np.ones( 4, dtype=int), np.zeros( 4, dtype=int)), axis=None)
        rt22 = np.concatenate((np.zeros(48, dtype=int), np.zeros(44, dtype=int), np.ones( 4, dtype=int)), axis=None)
        kt12 = np.concatenate((np.zeros(48, dtype=int), np.zeros(24, dtype=int), np.ones( 4, dtype=int), np.zeros(20, dtype=int)), axis=None)
        kt22 = np.concatenate((np.zeros(48, dtype=int), np.zeros(28, dtype=int), np.ones( 4, dtype=int), np.zeros(16, dtype=int)), axis=None)
        ck12 = np.concatenate((np.zeros(48, dtype=int), np.ones( 4, dtype=int), np.zeros(44, dtype=int)), axis=None)
        ck22 = np.concatenate((np.zeros(48, dtype=int), np.zeros( 4, dtype=int), np.ones( 4, dtype=int), np.zeros(40, dtype=int)), axis=None)
        cr12 = np.concatenate((np.zeros(48, dtype=int), np.zeros( 8, dtype=int), np.ones( 4, dtype=int), np.zeros(36, dtype=int)), axis=None)
        cr22 = np.concatenate((np.zeros(48, dtype=int), np.zeros(12, dtype=int), np.ones( 4, dtype=int), np.zeros(32, dtype=int)), axis=None)
        rk12 = np.concatenate((np.zeros(48, dtype=int), np.zeros(32, dtype=int), np.ones( 4, dtype=int), np.zeros(12, dtype=int)), axis=None)
        rk22 = np.concatenate((np.zeros(48, dtype=int), np.zeros(36, dtype=int), np.ones( 4, dtype=int), np.zeros( 8, dtype=int)), axis=None)
        oness= np.ones( 96, dtype=int)
        # enter data as numpy arrays
        # set of row indices
        I = range(96)

        # set of column indices
        J = range(96)

        # create a model instance
        model = ConcreteModel()

        # create x and y variables in the model
        model.x = Var(J)
        model.y11 = Var(within=Integers, bounds=(0,1))
        model.y21 = Var(within=Integers, bounds=(0,1))
        model.y31 = Var(within=Integers, bounds=(0,1))
        model.y41 = Var(within=Integers, bounds=(0,1))
        model.y51 = Var(within=Integers, bounds=(0,1))
        model.y61 = Var(within=Integers, bounds=(0,1))
        model.y71 = Var(within=Integers, bounds=(0,1))
        model.y81 = Var(within=Integers, bounds=(0,1))
        model.y91 = Var(within=Integers, bounds=(0,1))
        model.y101 = Var(within=Integers, bounds=(0,1))
        model.y12 = Var(within=Integers, bounds=(0,1))
        model.y22 = Var(within=Integers, bounds=(0,1))
        model.y32 = Var(within=Integers, bounds=(0,1))
        model.y42 = Var(within=Integers, bounds=(0,1))
        model.y52 = Var(within=Integers, bounds=(0,1))
        model.y62 = Var(within=Integers, bounds=(0,1))
        model.y72 = Var(within=Integers, bounds=(0,1))
        model.y82 = Var(within=Integers, bounds=(0,1))
        model.y92 = Var(within=Integers, bounds=(0,1))
        model.y102 = Var(within=Integers, bounds=(0,1))
        #model.y = Var(within=Integers, bounds=(0,1))

        # add model constraints
        model.constraints = ConstraintList()
        for i in I:
            model.constraints.add(sum(id[i,j]*model.x[j] for j in J) <= d[i])
        for i in I:
            model.constraints.add(sum(id[i,j]*model.x[j] for j in J) >= 0)

        model.constraints.add(sum(c11[j]*model.x[j] for j in J) <= 121)
        model.constraints.add(sum(c21[j]*model.x[j] for j in J) <= 412)
        model.constraints.add(sum(c31[j]*model.x[j] for j in J) <= 121)
        model.constraints.add(sum(c41[j]*model.x[j] for j in J) <= 412)
        model.constraints.add(sum(c51[j]*model.x[j] for j in J) <= 121)
        model.constraints.add(sum(c61[j]*model.x[j] for j in J) <= 412)

        model.constraints.add(sum(c12[j]*model.x[j] for j in J) <= 121)
        model.constraints.add(sum(c22[j]*model.x[j] for j in J) <= 412)
        model.constraints.add(sum(c32[j]*model.x[j] for j in J) <= 121)
        model.constraints.add(sum(c42[j]*model.x[j] for j in J) <= 412)
        model.constraints.add(sum(c52[j]*model.x[j] for j in J) <= 121)
        model.constraints.add(sum(c62[j]*model.x[j] for j in J) <= 412)



        model.constraints.add(sum(ct11[j]*model.x[j] for j in J) >= model.y11*sum(ct11[j]*d[j] for j in J))
        model.constraints.add(sum(ct21[j]*model.x[j] for j in J) >= model.y21*sum(ct21[j]*d[j] for j in J))
        model.constraints.add(sum(rt11[j]*model.x[j] for j in J) >= model.y31*sum(rt11[j]*d[j] for j in J))
        model.constraints.add(sum(rt21[j]*model.x[j] for j in J) >= model.y41*sum(rt21[j]*d[j] for j in J))
        model.constraints.add(sum(kt11[j]*model.x[j] for j in J) >= model.y51*sum(kt11[j]*d[j] for j in J))
        model.constraints.add(sum(kt21[j]*model.x[j] for j in J) >= model.y61*sum(kt21[j]*d[j] for j in J))
        model.constraints.add(sum(ck11[j]*model.x[j] for j in J) >= model.y71*sum(ck11[j]*d[j] for j in J))
        model.constraints.add(sum(ck21[j]*model.x[j] for j in J) >= model.y81*sum(ck21[j]*d[j] for j in J))
        model.constraints.add(sum(cr11[j]*model.x[j] for j in J) >= model.y91*sum(cr11[j]*d[j] for j in J))
        model.constraints.add(sum(cr21[j]*model.x[j] for j in J) >= model.y101*sum(cr21[j]*d[j] for j in J))

        model.constraints.add(sum(rt11[j]*model.x[j] for j in J) <= model.y11*sum(rt11[j]*d[j] for j in J))
        model.constraints.add(sum(rt21[j]*model.x[j] for j in J) <= model.y21*sum(rt21[j]*d[j] for j in J)) 
        model.constraints.add(sum(kt11[j]*model.x[j] for j in J) <= model.y31*sum(kt11[j]*d[j] for j in J))
        model.constraints.add(sum(kt21[j]*model.x[j] for j in J) <= model.y41*sum(kt21[j]*d[j] for j in J))
        model.constraints.add(sum(ck11[j]*model.x[j] for j in J) <= model.y51*sum(ck11[j]*d[j] for j in J))
        model.constraints.add(sum(ck21[j]*model.x[j] for j in J) <= model.y61*sum(ck21[j]*d[j] for j in J))
        model.constraints.add(sum(cr11[j]*model.x[j] for j in J) <= model.y71*sum(cr11[j]*d[j] for j in J))
        model.constraints.add(sum(cr21[j]*model.x[j] for j in J) <= model.y81*sum(cr21[j]*d[j] for j in J))
        model.constraints.add(sum(rk11[j]*model.x[j] for j in J) <= model.y91*sum(rk11[j]*d[j] for j in J))
        model.constraints.add(sum(rk21[j]*model.x[j] for j in J) <= model.y101*sum(rk21[j]*d[j] for j in J))


        model.constraints.add(sum(ct12[j]*model.x[j] for j in J) >= model.y12*sum(ct11[j]*d[j] for j in J))
        model.constraints.add(sum(ct22[j]*model.x[j] for j in J) >= model.y22*sum(ct21[j]*d[j] for j in J))
        model.constraints.add(sum(rt12[j]*model.x[j] for j in J) >= model.y32*sum(rt11[j]*d[j] for j in J))
        model.constraints.add(sum(rt22[j]*model.x[j] for j in J) >= model.y42*sum(rt21[j]*d[j] for j in J))
        model.constraints.add(sum(kt12[j]*model.x[j] for j in J) >= model.y52*sum(kt11[j]*d[j] for j in J))
        model.constraints.add(sum(kt22[j]*model.x[j] for j in J) >= model.y62*sum(kt21[j]*d[j] for j in J))
        model.constraints.add(sum(ck12[j]*model.x[j] for j in J) >= model.y72*sum(ck11[j]*d[j] for j in J))
        model.constraints.add(sum(ck22[j]*model.x[j] for j in J) >= model.y82*sum(ck21[j]*d[j] for j in J))
        model.constraints.add(sum(cr12[j]*model.x[j] for j in J) >= model.y92*sum(cr11[j]*d[j] for j in J))
        model.constraints.add(sum(cr22[j]*model.x[j] for j in J) >= model.y102*sum(cr21[j]*d[j] for j in J))

        model.constraints.add(sum(rt12[j]*model.x[j] for j in J) <= model.y12*sum(rt11[j]*d[j] for j in J))
        model.constraints.add(sum(rt22[j]*model.x[j] for j in J) <= model.y22*sum(rt21[j]*d[j] for j in J)) 
        model.constraints.add(sum(kt12[j]*model.x[j] for j in J) <= model.y32*sum(kt11[j]*d[j] for j in J))
        model.constraints.add(sum(kt22[j]*model.x[j] for j in J) <= model.y42*sum(kt21[j]*d[j] for j in J))
        model.constraints.add(sum(ck12[j]*model.x[j] for j in J) <= model.y52*sum(ck11[j]*d[j] for j in J))
        model.constraints.add(sum(ck22[j]*model.x[j] for j in J) <= model.y62*sum(ck21[j]*d[j] for j in J))
        model.constraints.add(sum(cr12[j]*model.x[j] for j in J) <= model.y72*sum(cr11[j]*d[j] for j in J))
        model.constraints.add(sum(cr22[j]*model.x[j] for j in J) <= model.y82*sum(cr21[j]*d[j] for j in J))
        model.constraints.add(sum(rk12[j]*model.x[j] for j in J) <= model.y92*sum(rk11[j]*d[j] for j in J))
        model.constraints.add(sum(rk22[j]*model.x[j] for j in J) <= model.y102*sum(rk21[j]*d[j] for j in J))


        model.constraints.add(sum(model.x[j] for j in J) - s >= 0)
        # add a model objective
        model.objective = Objective(expr = sum(obj[j]*model.x[j] for j in J), sense=maximize)

        # create a solver
        solver = pyomo.environ.SolverFactory('glpk')

        # solve
        solver.solve(model)
        results=[]
        for j in J:
            results.append(model.x[j].value)
        if Dir=='CASA VOYAGEURS ---> TANGER VILLE ':
            results=results[:48]
            results=np.array(results)
            temp=np.array(["CK","CR","CT","KT","RT"])
            results=np.reshape(results, (6,8))
            df = pd.DataFrame(results, columns = ['A0','A1','A2','A3','B0','B1','B2','B3'],index=["CK",'CR','CT','KT','RK','RT'])
            st.dataframe(df)                
            
        else:
            results=results[48:]
            results=np.array(results)
            temp=np.array(["KC","RC","TC","TK","TR"])
            results=np.reshape(results, (6,8))
            df = pd.DataFrame(results, columns = ['A0','A1','A2','A3','B0','B1','B2','B3'],index=["KC",'RC','TC','TK','KR','TR'])
            st.dataframe(df)
