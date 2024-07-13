#In[]
import time
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib import rc
import numpy as np
np.set_printoptions(suppress=True)
import os
from random import sample
import pandas as pd
import json
from copy import deepcopy, copy
# import qsimcirq
import seaborn as sns
from scipy.optimize import minimize
import cirq
# import stim
from scipy.linalg import expm
import warnings
# 忽略所有警告
from torch.optim import Adam
import multiprocessing
import sys
from qiskit import QuantumCircuit, transpile,QuantumRegister
from qiskit.transpiler import CouplingMap
#In[]
#--- Settings
I = np.array([[1,0],[0,1]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j, 0]])
Z = np.array([[1,0],[0,-1]])
H = (1/np.sqrt(2))*np.array([[1,1],[1,-1]])
X2p = (1/np.sqrt(2))*np.array([[1,-1j],[-1j,1]])
X2m = (1/np.sqrt(2))*np.array([[1,1j],[1j,1]])
Y2p = (1/np.sqrt(2))*np.array([[1,-1],[1,1]])
Y2m = (1/np.sqrt(2))*np.array([[1,1],[-1,1]])
S = np.array([[1,0],[0,1j]])
Paulis_ops = [cirq.I, cirq.X, cirq.Y, cirq.Z]
clifford_list = [I, X, Y, Z, 
X2p, Y2p, X2m, Y2m, 
X2p.dot(Y2p),X2p.dot(Y2m),
X2m.dot(Y2p),X2m.dot(Y2m),
 Y2p.dot(X2p),Y2p.dot(X2m),
 Y2m.dot(X2p),Y2m.dot(X2m), 
 X.dot(Y2p), X.dot(Y2m), 
 Y.dot(X2p), Y.dot(X2m), 
 X2p.dot(Y2p.dot(X2p)), X2m.dot(Y2p.dot(X2m)),
 X2m.dot(Y2p.dot(X2p)), X2m.dot(Y2m.dot(X2p))]
zero_rho = np.array([[1,0],[0,0]])
one_rho = np.array([[0,0],[0,1]])
jia_rho=np.array([[1,1],[1,1]])/2
eigen_ops = [np.array([1,0]), np.array([1,1])/np.sqrt(2), np.array([1,1j])/np.sqrt(2), np.array([1,0])]
safeshot = 10**10
# DATAPATH = "/home/inspur/Desktop/program/CNRVD"
DATAPATH="D:/data/CNRVD/data"
# DATAPATH="/public/home/acfyw4bb5h/program/gao_CNRVD"
cx_map = [[[0,0], [0,1], [3,2], [3,3]],
    [[1,1], [1,0], [2,3], [2,2]], 
    [[2,1], [2,0], [1,3], [1,2]],
    [[3,0], [3,1], [0,2], [0,3]]]
# datapath = "/home/inspur/Desktop/program/CNRVD/data"
datapath="D:/data/CNRVD/data"
# datapath="/public/home/acfyw4bb5h/program/gao_CNRVD/data"
dsim=cirq.DensityMatrixSimulator()
# sim=qsimcirq.QSimSimulator(qsim_options=qsimcirq.QSimOptions(cpu_threads=10))

MAX_CORE=80
INTERVAL_TIME=0.3
NOISE_SCALE=True 
ZNE_TYPE="exp"
# ZNE_TYPE="rid"

#--- noise parameter
sy23_1e=1.6*10**-3
sy23_2e=6*10**-3
sy23_2dur=20*10**-9
sy23_t1dur=np.mean(np.array([25,18.8,29.1,24.4,14.9,21.9,20.6,27,23.9,24.1,18,18.5,13.3,19,21.3,26,30.1,33.3,16.6,25.3,21.2,18.9,22.9,10.6,21,23.8,24.6,32,29.9,28.5,21,21,18.2,22.2,22.8,22,23,26.7,22.6,29.6,14.4,21.9,23.9,20.7,26.9,21.4,22.5,4.8,20.8,27.3,25.7,34.4,20.7,30,19.5,21.9,19.3,23.8,23.7,33.4,28.7,17,22.7,23.3,24.8,35.3,22.8,24.3,31.9,24.6]))*10**-6
sy23_decay=1-np.exp(-sy23_2dur/sy23_t1dur)
sy23_noise={'decay':sy23_decay,'2qgate':sy23_2e,'1qgate':sy23_1e}


def finite_shot(n,Ep,nshot):
    if n==1:
        A=Ep[0]
        bins=np.array([0,A,1])
        if nshot<=safeshot:
            xx=np.random.random_sample(nshot)
            yy,_=np.histogram(xx, bins=bins)
            yy=yy/sum(yy)
            Ashot=yy[0]
        else:
            nrounds=nshot//safeshot
            yy=np.zeros(2)
            try:
                for i in range(nrounds):
                    xx=np.random.random_sample(safeshot)
                    p,_=np.histogram(xx, bins=bins)
                    yy+=p
                xx=np.random.random_sample(nshot%safeshot)
                p,_=np.histogram(xx, bins=bins)
                yy+=p
                #yy=yy/sum(yy)
                Ashot=yy[0]
            except:
                print(Ep)
        return np.array(Ashot,1-Ashot)
    else:
        bins=np.zeros(2**n+1)
        for i in range(2**n):
            bins[i+1]=bins[i]+Ep[i]
        if nshot<=safeshot:
            xx=np.random.random_sample(nshot)
            readp,_=np.histogram(xx, bins=bins)
            readp=readp/sum(readp)
        else:
            nrounds=nshot//safeshot
            readp=np.zeros(2**n)
            for i in range(nrounds):
                xx=np.random.random_sample(safeshot)
                p,_=np.histogram(xx, bins=bins)
                readp+=p
            xx=np.random.random_sample(nshot%safeshot)
            p,_=np.histogram(xx, bins=bins)
            readp+=p
            readp=readp/sum(readp)
        return readp

def vec2rho(vec):
    return np.array(np.mat(vec).transpose().dot(np.mat(vec).conjugate()))

def rho2p(rho,nshot):
    Ep=np.real(np.diag(rho))
    if nshot==False:
        return Ep
    else:
        n=int(round(np.log2(len(Ep))))
        rp=finite_shot(n,Ep,nshot)
        return rp

def draw_mat(X):
    a,b=np.shape(X)
    plt.figure()
    mesh = plt.pcolormesh(range(b),range(a),np.flipud(X), vmin=-1, vmax=1)
    if a==b:
        plt.axis('equal')
    plt.colorbar(mesh)

def savedata(data,fname,cover_old):
    joinedpath=os.path.join(DATAPATH,fname)
    if os.path.isfile(joinedpath):
        if cover_old==False:
            data.to_csv(joinedpath,mode='a',header=False)
        else:
            data.to_csv(joinedpath)
    else:
        data.to_csv(joinedpath,mode='a')

def f(n, base, qn)->str:
    a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,'A','B','C','D','E','F']
    b = []
    while True:
        s = n//base
        y = n%base
        b = b + [y]
        if s == 0:
            break
        n = s
    string = ''.join([str(x) for x in b[::-1]])
    #将str补全
    while len(string) < qn:
        string = '0' + string
    return string

def get_Zobs(w, qn):
    string = f(w, 2, qn)
    if string[0] == "0":
        obs = cirq.unitary(Paulis_ops[0])
    else:
        obs = cirq.unitary(Paulis_ops[3])
    for i in range(qn-1):
        if string[1+i] == "0":
            obs = np.kron(obs, cirq.unitary(Paulis_ops[0]))
        else:
            obs = np.kron(obs, cirq.unitary(Paulis_ops[3]))
    return obs

def get_Pobs(w, qn):
    string = f(w, 4, qn)
    obs = cirq.unitary(Paulis_ops[int(string[0], base = 4)])
    for i in range(qn-1):
        obs = np.kron(obs, cirq.unitary(Paulis_ops[int(string[i+1], base = 4)]))
    return obs

def continuous_matmul(m_list):
    for i in range(len(m_list)):
        if i == 0:
            matrix = m_list[0]
        else:
            matrix = np.matmul(matrix, m_list[i])
    return matrix

def continuous_tensor(m_list):
    for i in range(len(m_list)):
        if i == 0:
            matrix = copy(m_list[0])
        else:
            matrix = np.kron(matrix, copy(m_list[i]))
    return matrix


def tensor_product(obs, times):
    #support vec or matrix
    matrix = obs.copy()
    for i in range(times-1):
        matrix = np.kron(matrix, obs.copy())
    return matrix

def vec_expectation(vec,obs):
    return np.real(np.mat(vec).conjugate().dot(np.mat(obs)).dot(np.mat(vec).transpose())[0,0])

def rho_expectation(rho,obs):
    return np.real(np.trace(obs.dot(rho)))

def probs_expectation(p,obs):
    a,b=np.shape(obs)
    flag=0
    assert a==b
    for i in range(a):
        for j in range(b):
            if i!=j:
                flag+=np.abs(obs[i,j])
    assert flag<1e-3
    d=np.diag(obs)
    return np.real(p.dot(d))

def compute_expectation(qn, popu, w):
    return probs_expectation(popu,get_Zobs(w,qn))

def intlog2(s):
    return int(round(np.log2(s)))

def partial_rho(rho,s):
    a,_=np.shape(rho)
    n=intlog2(a)
    subrho=np.reshape(cirq.partial_trace(np.reshape(rho,tuple([2 for i in range(2*n)])),list(range(s))),(2**s,2**s))
    return subrho

def threeGates_depo_rate(average_rate,num):
    p=1-average_rate/(1-1/4)
    depo_rate=p**int(num)
    pe3=(1-depo_rate)*(1-1/64)
    return pe3

#--- vec,state
def rand_vec(n,seed):
    rnd=np.random.RandomState(seed)
    random_phases=rnd.uniform(0, 2*np.pi, size=2**n)
    state_vector=np.zeros(2**n, dtype=np.complex64)
    state_vector+=np.exp(1.0j*random_phases)
    state_vector/=np.linalg.norm(state_vector)
    return state_vector

def rand_rho(n):
    return vec2rho(rand_vec(n))

def zero_vec(qn):
    a = np.zeros(2**qn)
    a[0] = 1
    return np.array(a)

def one_vec(n):
    a=np.zeros(2**n)
    a[-1]=1
    return np.array(a)

def ghz_vec(n):
    return (1/np.sqrt(2))*np.array([1 if i==0 or i==2**n-1 else 0 for i in range(2**n)])

def ghz_rho(n):
    return vec2rho(ghz_vec(n))

def w_vec(n):
    p = np.zeros((2**n,))
    for i in range(n):
        p[2**i] = 1/np.sqrt(n)
    return p

def w_rho(n):
    return vec2rho(w_vec(n))

def generate_Pauli_eigenvec(qn,obs):
    obs_string=f(obs,4,qn)
    return continuous_tensor([eigen_ops[int(s,base=4)] for s in obs_string])

def generate_Pauli_eigencirc(qn,obs):
    obs_string=f(obs,4,qn)
    circ=cirq.Circuit()
    q=cirq.LineQubit.range(qn)
    for i, s in enumerate(obs_string):
        if s=="1":
            circ.append(cirq.H(q[i]))
        elif s=="2":
            circ.append(cirq.H(q[i]))
            circ.append(cirq.S(q[i]))
    return circ

#random obs
def random_obs(qn,seed,weight=False):
    rnd=np.random.RandomState(seed)
    if weight==False:
        return rnd.randint(4**qn)
    else:
        chosen_idx=rnd.choice(range(qn),size=weight,replace=False)
        s=""
        idx=0
        for i in range(qn):
            if i in chosen_idx:
                s+=str(rnd.randint(1,4))
                idx+=1
            else:
                s+="0"
        return int(s,base=4)

def compute_weight(qn,obs):
    trivial_num=np.sum([1 if s=="0" else 0 for s in f(obs,4,qn)])
    return qn-trivial_num

#--- general add noise
def if_moment_have_multis(moment):
    for gate in moment:
        if len(gate.qubits)>=2:
            return True
    return False

def custom_key(s):
    if s.startswith('I') or s.startswith('Z'):
        return (0,s)
    else:
        return (1,s)
def generate_pauli_probs(qn,p0,seed):
    rnd=np.random.RandomState(seed)
    if qn==1:
        items=2
    else:
        items=6
    prob=rnd.random(items)
    prob/=prob.sum()
    prob*=p0
    prob=list(sorted(prob,reverse=True))
    if qn>1:
        higher=3
    else:
        higher=2
    strings=[idx2paulistring(random_obs(qn,seed+1000+i,weight=rnd.randint(1,higher)),qn) for i in range(items)]
    # if qn==3:
    #     strings = sorted(strings, key=custom_key)
    # print(strings)
    probabilities={k: v for k, v in zip(strings, prob)}
    # print(probabilities)
    return probabilities

def idx2paulistring(idx,qn):
    s=f(idx,4,qn)
    s=s.replace('0', 'I')
    s=s.replace('1', 'X')
    s=s.replace('2', 'Y')
    s=s.replace('3', 'Z')
    return s

def pauli_error(qn,p0,seed,ori_qn,order):
    if qn>1:
        probabilities=generate_pauli_probs(qn,p0,seed)
        return cirq.asymmetric_depolarize(error_probabilities=probabilities)
    else:
        return [cirq.asymmetric_depolarize(error_probabilities=generate_pauli_probs(qn,p0,seed+i+10)) for i in range(order*ori_qn+1)]

def add_noise(qn,circ:cirq.Circuit,noise_dict,noise_seed=0,ori=False,order=2):
    noisy_circuit = cirq.Circuit()
    q=cirq.LineQubit.range(qn)
    for gate in circ.all_operations():
        noisy_circuit.append(gate)

        for noise_type,rate in noise_dict.items():
            if ori:
                rate=1
            if noise_type=="seed":
                continue
            elif noise_type=="depo_each":
                single_depol_noise=cirq.depolarize(rate*sy23_1e)
                two_depol_noise=cirq.depolarize(rate*sy23_2e)
                if len(gate.qubits)==1:
                    noisy_circuit.append(single_depol_noise.on_each(gate.qubits))
                elif len(gate.qubits)>=2:
                    noisy_circuit.append(two_depol_noise.on_each(gate.qubits))
            elif noise_type=="depo":
                pd1=rate*sy23_1e
                pd2=((rate*sy23_2e)/(1-1/4))*(1-1/16)
                pd3=threeGates_depo_rate(rate*sy23_2e,6)
                single_depol_noise=cirq.depolarize(pd1)  
                two_depol_noise=cirq.depolarize(pd2,n_qubits=2)
                three_depol_noise=cirq.depolarize(pd3,n_qubits=3)
                if len(gate.qubits)==1:
                    noisy_circuit.append(single_depol_noise.on_each(gate.qubits))
                elif len(gate.qubits)==2:
                    noisy_circuit.append(two_depol_noise.on(*gate.qubits))
                elif len(gate.qubits)==3:
                    noisy_circuit.append(three_depol_noise.on(*gate.qubits))
            elif noise_type=="phase":
                single_phase_noise=cirq.phase_damp(rate*sy23_1e)
                two_phase_noise=cirq.phase_damp(rate*sy23_2e)
                if len(gate.qubits)==1:
                    noisy_circuit.append(single_phase_noise.on_each(gate.qubits))
                elif len(gate.qubits)>=2:
                    noisy_circuit.append(two_phase_noise.on_each(gate.qubits))
            elif noise_type=="decay":
                ampl_noise=cirq.amplitude_damp(rate*sy23_decay)
                if len(gate.qubits)==1:
                    noisy_circuit.append(ampl_noise.on_each(gate.qubits))
                elif len(gate.qubits)>=2:
                    noisy_circuit.append(ampl_noise.on_each(gate.qubits))
            elif noise_type=="pauli":
                pd2=((rate*sy23_2e)/(1-1/4))*(1-1/16)
                pd3=threeGates_depo_rate(rate*sy23_2e,6)
                pd1=rate*sy23_1e
                single_pauli_noise_list=pauli_error(1,pd1,noise_seed,qn,order)
                # print(single_pauli_noise)
                two_pauli_noise=pauli_error(2,pd2,noise_seed,qn,order)
                three_pauli_noise=pauli_error(3,pd3,noise_seed,qn,order)
                if len(gate.qubits)==1:
                    try:
                        noisy_circuit.append(single_pauli_noise_list[gate.qubits[0].x].on(*gate.qubits))
                    except:
                        raise Exception(f"{gate.qubits[0].x}")
                elif len(gate.qubits)==2:
                    noisy_circuit.append(two_pauli_noise.on(*gate.qubits))
                elif len(gate.qubits)==3:
                    noisy_circuit.append(three_pauli_noise.on(*gate.qubits))
    return noisy_circuit
#--- readout
def readjson(fname):
    # print(fname)
    joinedpath=os.path.join(datapath,fname)
    # print(joinedpath)
    try:
        with open(joinedpath, 'r') as f:
            data=f.read()
            data_dict = json.loads(data)
            return data_dict
    except:
        return 'fail'

def binlist2num(n,l):
    return sum([l[n-1-i]*2**i for i in range(n)])

def read0630_6():
    fname="20230630readoutdata/all.json"
    dict3=readjson(fname)['dataall']
    n=6
    T=np.zeros((2**n,2**n))
    for i in range(2**n):
        arr=dict3[i]
        for result in arr:
            lst = [1 if elem else 0 for elem in result]
            T[binlist2num(n,lst),i]+=1
    T=T/5000
    return T

def partial_tran(T, dimA):

    a=np.shape(T)[0]
    dimB=int(a/dimA)
    T = np.reshape(T, (dimA, dimB, dimA, dimB))
    b = np.zeros((dimA, dimA))
    for k in range(dimA):
        for i in range(dimA):
            s = 0
            for j in range(dimB):
                for l in range(dimB):
                    s += T[k,j,i,l]
            b[k,i] = s
    return b

def make_transition_mat(n):
    calied_n=6
    T=read0630_6()
    T=partial_tran(T, 2**n)/2**(calied_n-n)
    return T

def count_elements(arr, condition):
    count = 0
    for element in arr:
        if condition(element):
            count += 1
    return count

def calied_transition_submats(idx_list):
    idx_list=[i%6 for i in idx_list]
    qid=[2,5,6,9,10,11]
    submats=[]
    for i in idx_list:
        fname=f"20230630readoutdata/{qid[i]}.json"
        dict3=readjson(fname)
        dataall=dict3['dataall']
        p00=count_elements(dataall[0], lambda x: x ==[False])/len(dataall[0])
        p11=count_elements(dataall[1], lambda x: x ==[True])/len(dataall[1])
        T=np.array([[p00,1-p11],[1-p00,p11]])
        submats.append(T)
    return submats

def calied_transition_mat(idx_list):
    each_T=calied_transition_submats(idx_list)
    TT=1
    for i in range(len(idx_list)):
        TT=np.kron(TT,each_T[i])
    return np.real(TT)

def unfolding_round(qn,vec,cali_T,IBU_iteration):
    cur_vec = np.array([np.random.rand() for ii in range(2**qn)])
    for epoch in range(IBU_iteration):
        vlist = np.array([0.0 for i in range(2**qn)])
        for i in range(2**qn):
            v = 0
            for j in range(2**qn):
                deno = sum([cali_T[j,k]*cur_vec[k] for k in range(2**qn)])
                v += (cali_T[j,i]*cur_vec[i]*vec[j])/deno
            vlist[i] = v
        cur_vec = vlist.copy()
    return cur_vec

def query_T(qn):
    datafile_T=datapath+"/mem/T_{}.npy".format(qn)
    datafile_caliT=datapath+"/mem/caliT_{}.npy".format(qn)
    if os.path.exists(datafile_T):
        T=np.array(np.load(datafile_T))
    else:
        T=make_transition_mat(qn)
        np.save(datafile_T,T)
    if os.path.exists(datafile_caliT):
        cali_T=np.array(np.load(datafile_caliT))
    else:
        cali_T=calied_transition_mat(list(range(qn)))
        np.save(datafile_caliT,cali_T)
    op=lambda x: unfolding_round(qn,x,cali_T,50)
    return [np.array(T),op]

#--- general measure
def add_obs_basis_transfer(circ,q,qn,start,obs):
    obs_string = f(obs,4,qn)
    w = ""
    for i, s in enumerate(obs_string):
        if s == "1":
            circ.append(cirq.H(q[i+start]))
            w += "1"
        elif s == "2":
            circ.append((cirq.S**(-1))(q[i+start]))
            circ.append(cirq.H(q[i+start]))
            w += "1"
        elif s == "3":
            w += "1"
            circ.append(cirq.I(q[i+start]))
        else:
            w += "0"
            circ.append(cirq.I(q[i+start]))
    return circ,int(w,base=2)

def measure_all_density(qn,obs,nshots,initial_state,noise_dict,mem=False,circuit=None):
    #initial_state: state or density matrix
    q=cirq.LineQubit.range(qn)
    if circuit:
        circ=copy(circuit)
    else:
        circ=cirq.Circuit()
    circ,w=add_obs_basis_transfer(circ,q,qn,0,obs)
    if len(noise_dict)>0:
        if "seed" in noise_dict.keys():
            circ=add_noise(qn,circ,noise_dict,noise_seed=noise_dict["seed"])
        else:
            circ=add_noise(qn,circ,noise_dict)
    rho=dsim.simulate(circ,initial_state=initial_state).final_density_matrix
    if isinstance(nshots,float):
        nshots=int(nshots)
    #meaurement error
    if mem:
        T,op=query_T(qn)
        Ep=T.dot(np.real(np.diag(rho)))
        popu=op(Ep)
    else:
        popu = np.array(rho2p(rho,nshots))
    if nshots!=False:
        popu=finite_shot(qn,popu,nshots)
    if isinstance(popu,float):
        popu=np.array([popu,1-popu])
    # print(popu)
    obs_op=get_Zobs(w, qn)
    return probs_expectation(np.array(popu)/np.sum(popu),obs_op)

#--- add circuit
def add_pauli(circ,q,n,pidx):
    if pidx>0:
        ps=f(pidx,4,n)
        pauli_moment=[]
        for i,p in enumerate(ps):
            if p=="1":
                pauli_moment.append(cirq.X(q[i]))
            elif p=="2":
                pauli_moment.append(cirq.Y(q[i]))
            elif p=="3":
                pauli_moment.append(cirq.Z(q[i]))
        circ.append(cirq.Moment(pauli_moment))
    return circ

def CY():
    circ=cirq.Circuit()
    q=cirq.LineQubit.range(2)
    circ.append((cirq.S**(-1))(q[1]))
    circ.append(cirq.CX(q[0],q[1]))
    circ.append(cirq.S(q[1]))
    uni=cirq.unitary(circ)
    return cirq.MatrixGate(uni)

#--- VD-based measurement and circuit
def add_controlled_gate(circ,qubits,idx):
    if idx==1:
        circ.append(cirq.Moment(cirq.CX(*qubits)))
    elif idx==2:
        circ.append(cirq.Moment([(CY())(*qubits)]))
        # circ.append((cirq.S**(-1))(qubits[1]))
        # circ.append(cirq.CX(*qubits))
        # circ.append(cirq.S(qubits[1]))
    elif idx==3:
        circ.append(cirq.Moment(cirq.CZ(*qubits)))
    return circ 

def add_vd(circ,qn,obs,order,pidx_set):
    #先加一排
    q=cirq.LineQubit.range(order*qn+1)
    circ.append(cirq.H(q[0]))
    circ=add_pauli(circ,q,order*qn+1,pidx_set[0])  
    for order_idx in range(order-1):
        for i in range(1, qn+1):
            circ.append(cirq.Moment(cirq.CSWAP(q[0],q[i+order_idx*qn],q[i+(order_idx+1)*qn])))
    if obs > 0:
        circ=add_pauli(circ,q,order*qn+1,pidx_set[1])
        pauli_string=f(obs,4,qn)
        for i,s in enumerate(pauli_string):
            circ=add_controlled_gate(circ,[q[0],q[i+1]],int(s))
    circ.append(cirq.H(q[0]))
    return circ

def add_amplified_gates(ori_circ,amplified):
    new_circ=cirq.Circuit()
    for moment in ori_circ:
        new_circ.append(moment)
        if_amplify=False
        for gate in moment:
            if len(gate.qubits)>=2:
                if_amplify=True
                break
        if if_amplify:
            for t in range(amplified-1):
                new_circ.append(moment)
    return new_circ

def toVD_circ(qn,order,circ,simu_type):
    new_circ=cirq.Circuit()
    q=cirq.LineQubit.range(order*qn+1)
    # if simu_type=="count":
    #     new_circ.append(cirq.H(q[0]))
    for order_idx in range(order):
        new_circ.append(circ.transform_qubits(lambda q:q+1+order_idx*qn))
    return new_circ
    
def ancilla_measure(qn,order,simu_type,obs,nshots,initial_state,amplified=1,pidx_set=[0,0,0],noise_dict={},circuit=None,ori=False):
    #initial_state: statevec or density matrix
    # q=cirq.LineQubit.range(order*qn+1)
    vd_circ=add_vd(cirq.Circuit(),qn,obs,order,pidx_set)
    if amplified>1:
        # vd_circ=add_amplified_gates(vd_circ,amplified)
        noise_dict_ampli=copy(noise_dict)
        for k,v in noise_dict_ampli.items():
            if k=="seed":
                continue
            noise_dict_ampli[k]=0.94*amplified*v
    # vd_circ=compile_circuit(vd_circ,qn,order)
    if len(noise_dict)>0:
        if amplified>1:
            if NOISE_SCALE:
                if "seed" in noise_dict.keys():
                    vd_circ=add_noise(order*qn+1,copy(vd_circ),noise_dict_ampli,noise_seed=noise_dict["seed"],order=order)
                else:
                    vd_circ=add_noise(order*qn+1,copy(vd_circ),noise_dict_ampli,order=order)
            else:
                vd_circ=add_amplified_gates(vd_circ,amplified)
                if "seed" in noise_dict.keys():
                    vd_circ=add_noise(order*qn+1,copy(vd_circ),noise_dict,noise_seed=noise_dict["seed"],order=order)
                else:
                    vd_circ=add_noise(order*qn+1,copy(vd_circ),noise_dict,order=order)
        else:
            if "seed" in noise_dict.keys():
                vd_circ=add_noise(order*qn+1,copy(vd_circ),noise_dict,noise_seed=noise_dict["seed"],order=order)
            else:
                vd_circ=add_noise(order*qn+1,copy(vd_circ),noise_dict,order=order)
        if circuit:
            circuit=toVD_circ(qn,order,circuit,simu_type)
            if "seed" in noise_dict.keys():
                circuit=add_noise(qn,copy(circuit),noise_dict,noise_seed=noise_dict["seed"],order=order)
            else:
                circuit=add_noise(qn,copy(circuit),noise_dict,order=order)
    if circuit:
        # print(f(obs,4,qn))
        # print(circuit)
        # raise Exception
        circ=copy(circuit)+copy(vd_circ)
    else:
        circ=copy(vd_circ)
        
    # circ.append(cirq.H(q[0]))
    # return circ
    if isinstance(nshots,float):
        nshots=int(nshots)

    if simu_type=="density":
        if len(initial_state.shape)==2:
            ini_state=np.kron(zero_rho,tensor_product(initial_state,order))
            initial_state/=np.trace(initial_state)
        else:
            ini_state=np.kron(eigen_ops[0],tensor_product(initial_state,order))
        rho=np.array(dsim.simulate(circ,initial_state=ini_state).final_density_matrix)
        subrho=partial_rho(rho,1)
        popu=rho2p(subrho, nshots)
    if isinstance(popu,float):
        popu=np.array([popu,1-popu])
    elif isinstance(popu,np.ndarray) and popu.shape==():
        popu=np.array([popu,1-popu])
    if obs>0:
        pidx=pidx_set[2]
    else:
        pidx=pidx_set[0]
    if pidx!=0:
        pauli_string=f(pidx,4,order*qn+1)
        if pauli_string[0]=="3" or pauli_string[0]=="2":
            popu=list(reversed(popu))
    return popu[0]

#--- about twirling
def complete_pidx(w,n,target_n):
    string=f(w,4,n)
    for i in range(target_n-n):
        string = string+"0"
    return int(string,base=4)

def pauli_product(a,b):
    a=int(a,base=4)
    b=int(b,base=4)
    if a==b:
        return 0
    elif a==0:
        return b
    elif b==0:
        return a
    elif (a==1 and b==3) or (a==3 and b==1):
        return 2
    elif (a==2 and b==3) or (a==3 and b==2):
        return 1
    elif (a==1 and b==2) or (a==2 and b==1):
        return 3

def converge_pidx(n,pidx1,pidx2):
    ps1=f(pidx1,4,n)
    ps2=f(pidx2,4,n)
    new_ps=""
    for i in range(n):
        new_ps+=str(pauli_product(ps1[i],ps2[i]))
    return int(new_ps,base=4)   

def index2ps(qn,pidx):
    #pidx 2 cirq.paulistring
    q=cirq.LineQubit.range(qn)
    ps=f(pidx,4,qn)
    out=[]
    for i,s in enumerate(ps):
        out.append(Paulis_ops[int(s)](q[i]))
    return cirq.PauliString(out)

def ps2index(n,ps):
    idx_list=[]
    q=cirq.LineQubit.range(n)
    for i in range(n):
        if q[i] in ps.keys():
            idx_list.append(str(Paulis_ops.index(ps[q[i]])))
        else:
            idx_list.append(str(0))
    idx_string="".join(idx_list)
    return int(idx_string,base=4)

def compute_conjugate(qn,pidx,conjugate_circuit,order):
    P1=index2ps(qn+1, int(f(pidx,4,order*qn+1)[:qn+1],base=4))
    P2=P1.conjugated_by(conjugate_circuit)
    return complete_pidx(ps2index(qn+1,P2),qn+1,order*qn+1)

def random_cswap_pidx(qn,idx,order):
    ps=["{}".format(np.random.choice([0,3],1)[0])]+["0"]*(order*qn)
    chosen="{}".format(np.random.choice([0,1,2,3],1)[0])
    for order_idx in range(order):
        ps[order_idx*qn+idx+1]=chosen
    # print(ps)
    return int("".join(ps),base=4)

def create_vd_obs_circ(qn,obs):
    q=cirq.LineQubit.range(qn+1)
    obs_string=f(obs,4,qn)
    circ=cirq.Circuit()
    for i,s in enumerate(obs_string):
        circ=add_controlled_gate(circ,[q[0],q[i+1]],int(s,base=4))
    return circ

def random_list_vd(qn,order,random_instance,obs):
    obs_circ=create_vd_obs_circ(qn,obs)
    cswap_pidx_list=[]
    for i in range(random_instance*5):
        for ii in range(qn):
            if ii==0:
                ps=random_cswap_pidx(qn, ii,order)
            else:
                ps=converge_pidx(order*qn+1,ps,random_cswap_pidx(qn, ii,order))
        cswap_pidx_list.append(ps)
    cx_pidx_list=[complete_pidx(a,qn+1,order*qn+1) for a in np.random.randint(0,4**(qn+1),random_instance*5)]
    pidx_list=[]
    for i in range(random_instance*5):
        inter_idx=converge_pidx(order*qn+1, cswap_pidx_list[i], cx_pidx_list[i])
        cx_back_idx=compute_conjugate(qn,cx_pidx_list[i],obs_circ,order)
        pidx_list.append([cswap_pidx_list[i],inter_idx,cx_back_idx])
    pidx_list=list(np.unique(pidx_list,axis=0))
    np.random.shuffle(pidx_list)
    pidx_list=pidx_list[:random_instance]
    return pidx_list

#--- VD method preparation
#ZNE
def compute_gamma(n_points,idx):
    gamma=1
    for i, x in enumerate(n_points):
        if i!=idx:
            gamma*=x/(x-n_points[idx])
    return gamma
#SD
def generate_SWAP(qn):
    q=cirq.LineQubit.range(2*qn)
    circ=cirq.Circuit()
    for i in range(qn):
        circ.append(cirq.SWAP(q[i],q[i+qn]))
    return cirq.unitary(circ)

def shadow_computation(single_unitary,single_state):
    return 3*(single_unitary.conjugate().T).dot(single_state.dot(single_unitary))-np.eye(2)

clifford_zero_list=[shadow_computation(clifford_list[i],zero_rho) for i in range(len(clifford_list))]
clifford_one_list=[shadow_computation(clifford_list[i],one_rho) for i in range(len(clifford_list))]

def query_S2(qn):
    datafile=datapath+"/sd/S_{}".format(qn)
    if os.path.exists(datafile):
        return np.array(np.load(datafile))
    else:
        S2=generate_SWAP(qn)
        np.save(datafile,S2)
        return S2
#In[]
def compute_one_basis_shadow(qn, Ns, popu, basis):
    popu=np.array(popu)
    popu/=popu.sum()
    #popu中所有非0项
    chosen_bit=np.random.choice(range(2**qn),size=Ns,p=popu)
    basis_rho=np.zeros((2**qn,2**qn),dtype=np.complex128)
    clifford_outcome_dict={"0":clifford_zero_list,"1":clifford_one_list}
    # print(includsing_index[0])
    for idx in chosen_bit:
        outcome_rho=continuous_tensor([clifford_outcome_dict[s][basis[i]-1] for i,s in enumerate(f(idx,2,qn))])
        # print(np.real(np.trace(outcome_rho)))
        basis_rho+=np.array(outcome_rho,dtype=np.complex128)
    # print(np.real(np.trace(basis_rho)))
    return basis_rho/Ns

def compute_basis_list_shadow(qn,basis_list,popu_list,Ns):
    Nu=len(basis_list)
    shadow_list=[compute_one_basis_shadow(qn, Ns, popu_list[i], basis_list[i]) for i in range(Nu)]
    return shadow_list

def fast_all(shadow_list,obs):
    return np.dot((np.sum(shadow_list, axis=0)),obs)

def fast_kron(qn,shadow_list,all_deno_shadow,obs):
    a=np.zeros((4**qn,4**qn),dtype='complex128')
    for shadow in shadow_list:
        shadow=shadow-np.zeros((2**qn,2**qn),dtype='complex128')
        a+=np.kron(shadow,all_deno_shadow-np.dot(shadow,obs))
    return a

def fast_dot(qn,Nu,shadow_pre_deno):
    S2=query_S2(qn)
    return np.real(np.trace(np.dot(S2, shadow_pre_deno)))/(Nu*(Nu-1))

def shadow_estimation(qn,basis,initial_state,noise_dict,mem=False,circuit=None):
    q=cirq.LineQubit.range(qn)
    if circuit:
        circ=copy(circuit)
    else:
        circ=cirq.Circuit()
    if len(noise_dict)>0:
        circ=add_noise(qn,circ,noise_dict)
    #add SD basis
    SD_op_list=[cirq.MatrixGate(clifford_list[basis[i]-1]).on(q[i]) for i in range(qn)]
    circ.append(cirq.Moment(SD_op_list))
    rho=dsim.simulate(circ,initial_state=initial_state).final_density_matrix
    #noise process
    Ep=np.real(np.diag(rho))
    if mem:
        T,op=query_T(qn)
        Ep=T.dot(Ep)
        Ep=op(Ep)
    Ep=[0 if abs(a)<1e-6 else a for a in Ep]
    Ep=Ep/sum(Ep)
    return Ep

def SD_expectation(qn,shadow_list,obs):
    # shadow_list = np.array([shadow_tuple[i][1] for i in range(len(shadow_tuple))])
    Nu=len(shadow_list)
    ob_op=get_Pobs(obs,qn)
    OO=np.array(ob_op, dtype="complex128")
    all_no_shadow=fast_all(shadow_list, OO)
    shadow_pre_no=fast_kron(qn,shadow_list,all_no_shadow,OO)
    exp=fast_dot(qn,Nu,shadow_pre_no)
    return exp

#--- VD-based algorithm
def query_VD(qn,obs,initial_state,order,nshots,random_instance,noise_dict,amplified_list=[1],circuit=None,circ_complie=False):
    if random_instance:
        pidx_list=random_list_vd(qn, order, random_instance,obs)
    else:
        pidx_list=[[0,0,0]]
    exp_noisy_list=[]
    for amplified in amplified_list:
        exp_list=[[],[]]
        for i,ob in enumerate([obs,0]):
            for pidx_set in pidx_list:
                if order*qn+1>13:
                    q0_noisy=ancilla_measure(qn,order,"count",ob,nshots,initial_state,amplified=amplified,pidx_set=pidx_set,noise_dict=noise_dict,circuit=circuit)
                else:
                    q0_noisy=ancilla_measure(qn,order,"density",ob,nshots,initial_state,amplified=amplified,pidx_set=pidx_set,noise_dict=noise_dict,circuit=circuit)
                exp_list[i].append(2*q0_noisy-1)
        exp_noisy_list.append([np.mean(exp_list[0]),np.mean(exp_list[1])])
    return exp_noisy_list

def noisyVD(qn,obs,initial_state,order,nshots,noise_dict={},circuit=None):
    noisy_rho_exp=query_VD(qn,obs,initial_state,order,nshots,0,noise_dict,circuit=circuit)[0]
    return noisy_rho_exp[0]/noisy_rho_exp[1]

def CNR_VD(qn,obs,initial_state,order,nshots,random_instance,noise_dict={},cali_value=False,circuit=None):
    noisy_rho_exp=query_VD(qn,obs,initial_state,order,nshots,random_instance,noise_dict,circuit=circuit)[0]
    noisy_exp=noisy_rho_exp[0]/noisy_rho_exp[1]
    if not cali_value:
        # if order*qn+1=<11:
        #     ini_state=generate_Pauli_eigenvec(qn,obs)
        #     ini_circuit=None
        # else:
        ini_state=zero_vec(qn)
        ini_circuit=generate_Pauli_eigencirc(qn,obs)
        cali_values=query_VD(qn,obs,ini_state,order,nshots,random_instance,noise_dict,circuit=ini_circuit,circ_complie=True)[0]
        # print(cali_values)
        cali_value=cali_values[0]/cali_values[1]
    # print(noisy_exp,cali_value)
    return noisy_exp/cali_value

def ZNE_VD(qn,obs,initial_state,order,nshots,random_instance,noise_dict={},circuit=None):
    amplified_list=[1,3]
    noisy_rho_exp_list=query_VD(qn,obs,initial_state,order,nshots,random_instance,noise_dict,amplified_list=amplified_list,circuit=circuit)
    if ZNE_TYPE=="rid":
        time1=noisy_rho_exp_list[0][0]/noisy_rho_exp_list[0][1]
        time3=noisy_rho_exp_list[1][0]/noisy_rho_exp_list[1][1]
        exp=time1*compute_gamma([1,3],0)+time3*compute_gamma([1,3],1)
    else:
        time1=noisy_rho_exp_list[0][0]/noisy_rho_exp_list[0][1]
        sign=np.sign(time1)
        time3=noisy_rho_exp_list[1][0]/noisy_rho_exp_list[1][1]
        exp=sign*(np.abs((time1)**3/time3)**(1/2))
    return exp

def SD(qn,obs,Nu,Ns,initial_state,noise_dict={},mem=False,circuit=None):
    exp_list=[]
    for ob in [obs,0]:
        #generate_shadows
        basis_list=[[np.random.randint(1,25) for j in range(qn)] for i in range(Nu)]
        popu_list=[shadow_estimation(qn,basis,copy(initial_state),noise_dict,mem=mem,circuit=circuit) for basis in basis_list]
        shadow_list=compute_basis_list_shadow(qn, basis_list, popu_list, Ns)
        exp_list.append(SD_expectation(qn,shadow_list,ob))
        # print(exp_list)
    return exp_list[0]/exp_list[1]

#--- control group 
def Unmit(qn,obs,initial_state,nshots,noise_dict={},mem=False,circuit=None):
    return measure_all_density(qn,obs,nshots,initial_state,noise_dict,mem=mem,circuit=circuit)

def Ideal(qn,obs,initial_state,circuit=None):
    q=cirq.LineQubit.range(qn)
    if circuit:
        circ=copy(circuit)
    else:
        circ=cirq.Circuit([cirq.I(q[i]) for i in range(qn)])
    rho=dsim.simulate(circ,initial_state=initial_state).final_density_matrix
    obs_op=get_Pobs(obs,qn)
    return np.real(np.trace(rho@obs_op))

def idealVD(qn,obs,initial_state,order,circuit=None):
    if circuit:
        rho=dsim.simulate(circuit,initial_state=initial_state).final_density_matrix
        distill_state=continuous_matmul([rho]*order)
    else:
        distill_state=continuous_matmul([initial_state]*order)
    obs_op=get_Pobs(obs,qn)
    nomi=np.real(np.trace(distill_state@obs_op))
    denomi=np.real(np.trace(distill_state))
    return nomi/denomi

#--- Experimental settings
def circ_to_state(qn,circ):
    return dsim.simulate(circ,initial_state=zero_vec(qn)).final_density_matrix

#random state
def generate_orthogonal_part(qn,seed):
    dominant=rand_vec(qn,seed)
    errornous=orthogonalize(rand_vec(qn,seed+1000), dominant)
    return vec2rho(dominant), vec2rho(errornous)

def orthogonalize(vector, basis):
    random_orthogonal_vector = vector - np.dot(vector, np.conj(basis)) / np.dot(basis, np.conj(basis)) * basis
    random_orthogonal_vector /= np.linalg.norm(random_orthogonal_vector)
    return random_orthogonal_vector

def random_state_purity(qn,p0,seed):
    p1=1-p0
    dominant, errornous=generate_orthogonal_part(qn,seed)
    return p0*dominant+p1*errornous, dominant


#--- Experiments-single task
def task_fig1(input_params):
    """
    settings：dict={"purity":must,"p0":must,"weight":must,"seed":must,"random_instance":chosen}
    """
    qn,order,obs,noise_dict,filename,mem,settings,nshots=input_params
    p0=settings["p0"]
    weight=settings["weight"]
    seed=settings["seed"]
    if "pauli" in noise_dict.keys():
        noise_dict["seed"]=seed
    if "random_instance" in settings.keys():
        random_instance=settings["random_instance"]
    else:
        random_instance=0
    data_save_file=datapath+"/{}".format(filename)
    initial_state,target_state=random_state_purity(qn,p0,seed)
    # obs=random_obs(qn,weight=weight)
    if weight==False:
        weight=qn
    unmit=Unmit(qn,obs,copy(initial_state),nshots,noise_dict=noise_dict,mem=mem)
    ideal=Ideal(qn,obs,copy(target_state))
    ideal_vd=idealVD(qn,obs,copy(initial_state),order)
    noisy_vd=noisyVD(qn,obs,copy(initial_state),order,nshots,noise_dict=noise_dict)
    cnr_vd=CNR_VD(qn,obs,copy(initial_state),order,nshots,random_instance,noise_dict=noise_dict)
    rate=noise_dict.values()
    data=pd.DataFrame({"qn":[str(qn)],"order":[str(order)],"seed":[str(seed)],"weight":[str(weight)],"obs":[str(obs)],"error_rate":[str(list(rate)[0])],"p0":[str(p0)],"random_instance":[str(random_instance)],"ideal":[str(ideal)],"noisy":[str(unmit)],"ideal_vd":[str(ideal_vd)],"noisy_vd":[str(noisy_vd)],"cnr_vd":[str(cnr_vd)]})
    savedata(data,data_save_file,cover_old=False)
 
#--- Experiments: Specific tasks

def order_exponential(qn,orders,weight,seeds,noise_dict,mem,p0,repetition_times=1,random_instance=0):
    filename="maintext/order_expo.csv"
    total_task_num=len(orders)*len(seeds)*repetition_times

    qn_list=[qn]*total_task_num  
    noise_dict_list=[noise_dict]*total_task_num
    filname_list=[filename]*total_task_num
    mem_list=[mem]*total_task_num
    nshots_list=[False]*total_task_num
    
    order_list=[] 
    settings_list=[]
    obs_list=[]
    for seed in seeds:
        obs=random_obs(qn,seed,weight=weight)
        for order in orders:
            setting={"seed":seed,"weight":weight,"p0":p0}
            if random_instance:
                setting["random_instance"]=random_instance
            settings_list.append(setting)
            order_list.append(order)
            obs_list.append(obs)
    settings_list=settings_list*(len(orders)*repetition_times)
    params_list=list(zip(qn_list,order_list,obs_list,noise_dict_list,filname_list,mem_list,settings_list,nshots_list))

    for param in params_list:
        task_fig1(param)

def noise_level(qn,weight,seeds,noise_type,noise_level_list,mem,p0,repetition_times=1,random_instance=0,order=2,task_id=0):
    if task_id>0:
        filename="maintext/order_expo.csv"
    else:
        filename="maintext/noise_threshold.csv"
    total_task_num=len(noise_level_list)*len(seeds)*repetition_times

    qn_list=[qn]*total_task_num  
    filname_list=[filename]*total_task_num
    mem_list=[mem]*total_task_num
    nshots_list=[False]*total_task_num
    order_list=[order]*total_task_num
    
    settings_list=[]
    noise_dict_list=[]
    obs_list=[]
    for seed in seeds:
        obs=random_obs(qn,seed,weight=weight)
        for rate in noise_level_list:
            noise_dict={noise_type:rate}
            setting={"seed":seed,"weight":weight,"p0":p0}
            if random_instance:
                setting["random_instance"]=random_instance
            settings_list.append(setting)
            noise_dict_list.append(noise_dict)
            obs_list.append(obs)

    settings_list=settings_list*repetition_times
    params_list=list(zip(qn_list,order_list,obs_list,noise_dict_list,filname_list,mem_list,settings_list,nshots_list))

    for param in params_list:
        task_fig1(param)
    # a=time.time()
    # pool=multiprocessing.Pool(processes=20)
    # for params in params_list:
    #     pool.apply_async(task_fig1, (params,))
    # pool.close()
    # pool.join()
    # b=time.time()
    # print(b-a)


#In[]
if __name__=="__main__":
    #fig1-order
    # mem=True
    # qn1=3
    # orders1=[2,3]
    # weight1=qn1
    # for p0 in p0_list:
    #     order_exponential(qn1,orders1,weight1,seeds,noise_dict,mem,p0)
    # print("fig1-order over!")

    #fig1-noise level
    qn2=4
    mem=True
    order2=2
    weight2=1
    # weight2=qn
    nls=[0.05,0.1,0.25,0.5,0.75,1,2.5,5,10,25]
    p0s=[0.7,0.75,0.8,0.85,0.9,0.95]
    seed=int(sys.argv[2])
    seeds=[a for a in range(seed,seed+10)]
    for nl in nls:
        for p0 in p0s:
            noise_level(qn2,weight2,seeds,"pauli",[nl],mem,p0,order=order2,task_id=1)



    
