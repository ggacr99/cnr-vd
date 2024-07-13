#In[]:
import numpy as np
np.set_printoptions(suppress=True)
import os
import sys
import warnings
# 过滤掉DeprecationWarning警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import json
import cirq
from copy import deepcopy,copy
from qiskit import QuantumCircuit, ClassicalRegister,Aer,transpile
from qiskit.transpiler import CouplingMap
from qiskit.circuit.library.standard_gates import CXGate,CYGate,CZGate
from qiskit.providers.aer.noise import NoiseModel,depolarizing_error,amplitude_damping_error,pauli_error
from qiskit.quantum_info import Operator
from qiskit.opflow import I as sigma_i
from qiskit.opflow import X as sigma_x
from qiskit.opflow import Y as sigma_y
from qiskit.opflow import Z as sigma_z
import time

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
safeshot=10**10
Paulis_ops=[cirq.I,cirq.X,cirq.Y,cirq.Z]
# DATAPATH="D:/data/CNRVD"
# DATAPATH="/public/home/acfyw4bb5h/program/gao_CNRVD"
cx_map=["00", "01", "32", "33",
    "11", "10", "23", "22", 
    "21", "20", "13", "12",
    "30", "31", "02", "03"]
cy_map=['30','01','32','03', 
    '22', '13', '20', '11', 
    '12', '23', '10', '21', 
    '00', '31', '02', '33']
cz_map=['00','31','32','03',
    '13', '22', '21', '10', 
    '23', '12', '11', '20', 
    '30', '01', '02', '33']
# datapath = "/home/inspur/Desktop/program/CNRVD/data"
datapath="D:/data/CNRVD"
# datapath="/public/home/acfyw4bb5h/program/gao_CNRVD/data"
NOISE_SCALE=False  #如果是False则使用gate unfolding
ZNE_TYPE="rid"

#--- noise parameter
sy23_1e=1.6*10**-3
sy23_2e=6*10**-3
sy23_2dur=20*10**-9
sy23_t1dur=np.mean(np.array([25,18.8,29.1,24.4,14.9,21.9,20.6,27,23.9,24.1,18,18.5,13.3,19,21.3,26,30.1,33.3,16.6,25.3,21.2,18.9,22.9,10.6,21,23.8,24.6,32,29.9,28.5,21,21,18.2,22.2,22.8,22,23,26.7,22.6,29.6,14.4,21.9,23.9,20.7,26.9,21.4,22.5,4.8,20.8,27.3,25.7,34.4,20.7,30,19.5,21.9,19.3,23.8,23.7,33.4,28.7,17,22.7,23.3,24.8,35.3,22.8,24.3,31.9,24.6]))*10**-6
sy23_decay=1-np.exp(-sy23_2dur/sy23_t1dur)
sy23_noise={'decay':sy23_decay,'2qgate':sy23_2e,'1qgate':sy23_1e}

#--- Basic function
def vec2rho(vec):
    return np.array(np.mat(vec).transpose().dot(np.mat(vec).conjugate()))

def rand_vec(n,seed):
    rnd=np.random.RandomState(seed)
    random_phases=rnd.uniform(0,2*np.pi,size=2**n)
    state_vector=np.zeros(2**n,dtype=np.complex64)
    state_vector+=np.exp(1.0j*random_phases)
    state_vector/=np.linalg.norm(state_vector)
    return state_vector

def threeGates_depo_rate(average_rate,num):
    p=1-average_rate/(1-1/4)
    depo_rate=p**int(num)
    pe3=(1-depo_rate)*(1-1/64)
    return pe3

pd3=threeGates_depo_rate(sy23_2e,6)

def int2string(n,base,qn)->str:
    b=[]
    while True:
        s=n//base
        y=n%base
        b=b+[y]
        if s == 0:
            break
        n=s
    string=''.join([str(x) for x in b[::-1]])
    while len(string)<qn:
        string='0'+string
    return string

def int2Pauli_string(n,base,qn):
    ps=int2string(n,base,qn)
    pauli_string=""
    for s in ps:
        if s=="0":
            pauli_string+="I"
        elif s=="1":
            pauli_string+="X"
        elif s=="2":
            pauli_string+="Y"
        else:
            pauli_string+="Z" 
    return pauli_string

def random_obs(qn,weight,base=4,seed=-1):
    if seed==-1:
        seed=np.random.randint(0,10000)
    rnd=np.random.RandomState(seed)
    if weight==False:
        pidx=rnd.randint(1,4**qn)
        s=""
        for ps in int2string(pidx,4,qn):
            if ps=="0":
                s+="I"
            elif ps=="1":
                s+="X"
            elif ps=="2":
                s+="Y"
            else:
                s+="Z"
    else:
        if weight>qn:
            weight=qn
        chosen_idx=rnd.choice(range(qn),size=weight,replace=False)
        if base==4:
            chosen_string=[rnd.choice(["X","Y","Z"],size=1)[0] for i in range(weight)]
        else:
            chosen_string=["Z" for i in range(weight)]
        s=""
        idx=0
        for i in range(qn):
            if i in chosen_idx:
                s+=chosen_string[idx]
                idx+=1
            else:
                s+="I"
    return s

def compute_weight(obs):
    weight=0
    for s in obs:
        if s=="X" or s=="Y" or s=="Z":
            weight+=1
    return weight

def savedata(data,fname,cover_old):
    joinedpath=fname
    if os.path.isfile(joinedpath):
        if cover_old==False:
            data.to_csv(joinedpath,mode='a',header=False)
        else:
            data.to_csv(joinedpath)
    else:
        data.to_csv(joinedpath,mode='a')

def get_Pobs(w,qn):
    string=int2string(w,4,qn)
    Pauli_ops=[I,X,Y,Z]
    obs=Pauli_ops[int(string[0],base=4)]
    for i in range(qn-1):
        obs=np.kron(obs,Pauli_ops[int(string[i+1],base=4)])
    return obs

def get_Zobs(w,qn):
    string=int2string(w,2,qn)
    Pauli_ops=[I,Z]
    obs=Pauli_ops[int(string[0],base=4)]
    for i in range(qn-1):
        obs=np.kron(obs,Pauli_ops[int(string[i+1],base=4)])
    return obs

def continuous_tensor(m_list):
    for i in range(len(m_list)):
        if i==0:
            matrix=copy(m_list[0])
        else:
            matrix=np.kron(matrix,copy(m_list[i]))
    return matrix

def tensor_product(obs,times):
    #support vec or matrix
    matrix=obs.copy()
    for i in range(times-1):
        matrix=np.kron(matrix,obs.copy())
    return matrix

def continuous_matmul(m_list):
    for i in range(len(m_list)):
        if i==0:
            matrix=m_list[0]
        else:
            matrix=np.matmul(matrix,m_list[i])
    return matrix

#--- Circuit
def string2Pauliop(pauli_string):
    pauli_op_dict={"I":sigma_i,"X":sigma_x,"Y":sigma_y,"Z":sigma_z}
    pauli_op=None
    for i in range(len(pauli_string)):
        if pauli_string[i]=="I":
            continue
        else:
            if not pauli_op:
                pauli_op=pauli_op_dict[pauli_string[i]]
            else:
                pauli_op=pauli_op^pauli_op_dict[pauli_string[i]]
    return pauli_op

def add_obs_basis(circ,qn,pauli:int):
    #all basis meas
    obs_string=int2string(pauli,4,qn)
    w=""
    new_circ=circ.copy()
    for i in range(qn):
        if obs_string[i]=='1':
            new_circ.h(i)
            w+="1"
        elif obs_string[i]=='2':
            new_circ.sdg(i)
            new_circ.h(i)
            w+="1"
        elif obs_string[i]=='3':
            w+="1"
        else:
            w+="0"
    return new_circ,int(w,base=2)

def add_single_pauli(circ,idx,ps):
    if ps!="0":
        if ps=="1":
            circ.x(idx)
        elif ps=="2":
            circ.y(idx)
        elif ps=="3":
            circ.z(idx)
    return circ

def add_twirling_cz(circ,i,j,pidx):
    pauli_s=int2string(pidx,4,2)
    if pidx>0:
        circ=add_single_pauli(circ,i,pauli_s[0])
        circ=add_single_pauli(circ,j,pauli_s[1])
    circ.cz(i,j)
    if pidx>0:
        new_pauli_s=cz_map[int(pauli_s,base=4)]
        circ=add_single_pauli(circ,i,new_pauli_s[0])
        circ=add_single_pauli(circ,j,new_pauli_s[1])
    return circ

def count_multi_gate_num(circ):
    ntq=0
    for instr,qargs,_ in circ.data:
        if len(qargs)>1 and instr.name!='barrier': 
            ntq+=1
    return ntq

#--- ZNE
def amplify_circ(circ,times,unfolding_type="gate"):
    new_circ=QuantumCircuit(circ.num_qubits,circ.num_qubits)
    for instr,qargs,cargs in circ.data:
        if len(qargs)>1: 
            if unfolding_type=="gate":
                for i in range(times):
                    new_circ.cz(qargs[0],qargs[1])
                    if i<times-1:
                        new_circ.barrier([qargs[0],qargs[1]])
            else:
                new_circ.append(instr,qargs,cargs)
        else:
            new_circ.append(instr,qargs,cargs)
    return new_circ

def compute_gamma(n_points,idx):
    gamma=1
    for i, x in enumerate(n_points):
        if i!=idx:
            gamma*=x/(x-n_points[idx])
    return gamma

#--- readout
def readjson(fname):
    # print(fname)
    joindpath=os.path.join(datapath,fname)
    # print(joinedpath)
    try:
        with open(joindpath,'r') as f:
            data=f.read()
            data_dict=json.loads(data)
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

def partial_tran(T,dimA):
    a=np.shape(T)[0]
    dimB=int(a/dimA)
    T=np.reshape(T,(dimA,dimB,dimA,dimB))
    b=np.zeros((dimA,dimA))
    for k in range(dimA):
        for i in range(dimA):
            s=0
            for j in range(dimB):
                for l in range(dimB):
                    s+=T[k,j,i,l]
            b[k,i]=s
    return b

def make_transition_mat(n):
    calied_n=6
    T=read0630_6()
    T=partial_tran(T,2**n)/2**(calied_n-n)
    return T

def count_elements(arr,condition):
    count=0
    for element in arr:
        if condition(element):
            count+=1
    return count

def calied_transition_submats(idx_list):
    idx_list=[i%6 for i in idx_list]
    qid=[2,5,6,9,10,11]
    submats=[]
    for i in idx_list:
        fname=f"20230630readoutdata/{qid[i]}.json"
        dict3=readjson(fname)
        dataall=dict3['dataall']
        p00=count_elements(dataall[0],lambda x:x ==[False])/len(dataall[0])
        p11=count_elements(dataall[1],lambda x:x ==[True])/len(dataall[1])
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
    cur_vec=np.array([np.random.rand() for _ in range(2**qn)])
    for _ in range(IBU_iteration):
        vlist=np.array([0.0 for i in range(2**qn)])
        for i in range(2**qn):
            v=0
            for j in range(2**qn):
                deno=sum([cali_T[j,k]*cur_vec[k] for k in range(2**qn)])
                v+=(cali_T[j,i]*cur_vec[i]*vec[j])/deno
            vlist[i]=v
        cur_vec=vlist.copy()
    return cur_vec

def read_T_cali(n):
    datafile_T=datapath+"/mem/T_{}.npy".format(n)
    datafile_caliT=datapath+"/mem/caliT_{}.npy".format(n)
    if os.path.exists(datafile_T):
        T=np.array(np.load(datafile_T))
    else:
        T=make_transition_mat(n)
        np.save(datafile_T,T)
    if os.path.exists(datafile_caliT):
        cali_T=np.array(np.load(datafile_caliT))
    else:
        cali_T=calied_transition_mat(list(range(n)))
        np.save(datafile_caliT,cali_T)
    return T,cali_T

def query_T(qn):
    if qn>6:
        n=6
    else:
        n=qn
    T,cali_T=read_T_cali(n)
    if qn>6:
        sup_T,sup_cali_T=read_T_cali(qn-n)
        T=np.kron(T,sup_T)
        cali_T=np.kron(cali_T,sup_cali_T)
    op=lambda x: unfolding_round(qn,x,cali_T,50)
    return [np.array(T),op]

#--- measure
def gen_noise(noise_model,qn,error_list,fluc_rate,seed):
    gate_dict={1:["u1","u2","u3"],2:["cx","cy","cz","swap"],3:["cswap"]}
    rnd=np.random.RandomState(seed)
    for i,(error_type,rate) in enumerate(error_list):
        if error_type=="depo":
            error=depolarizing_error(rate+fluc_rate*rnd.rand(),1)
            if qn==2:
                error_gate=depolarizing_error(rate+fluc_rate*rnd.rand(),1)
                error=error_gate.tensor(error_gate)
            elif qn==3:
                error_gate=depolarizing_error(rate+fluc_rate*rnd.rand(),1)
                error_gate2=error_gate.tensor(error_gate)
                error=error_gate.tensor(error_gate2)
        elif error_type=="phase_flip":
            error=pauli_error([('Z',rate+fluc_rate*rnd.rand())])
            if qn==2:
                error_gate=pauli_error([('Z',rate+fluc_rate*rnd.rand())])
                error=error_gate.tensor(error_gate)
            elif qn==3:
                error_gate=pauli_error([('Z',rate+fluc_rate*rnd.rand())])
                error_gate2=error_gate.tensor(error_gate)
                error=error_gate.tensor(error_gate2)
        elif error_type=="bias_pauli":
            px=rate*0.1+fluc_rate*rnd.rand()
            py=rate*0.1+fluc_rate*rnd.rand()
            pz=rate*0.8+fluc_rate*rnd.rand()
            error=pauli_error([("I",1-px-py-pz),('X',px),('Y',py),('Z',pz)])
            if qn==2:
                px=rate*0.1+fluc_rate*rnd.rand()
                py=rate*0.1+fluc_rate*rnd.rand()
                pz=rate*0.8+fluc_rate*rnd.rand()
                error_gate=pauli_error([("I",1-px-py-pz),('X',px),('Y',py),('Z',pz)])
                error=error_gate.tensor(error_gate)
            elif qn==3:
                px=rate*0.1+fluc_rate*rnd.rand()
                py=rate*0.1+fluc_rate*rnd.rand()
                pz=rate*0.8+fluc_rate*rnd.rand()
                error_gate=pauli_error([("I",1-px-py-pz),('X',px),('Y',py),('Z',pz)])
                error_gate2=error_gate.tensor(error_gate)
                error=error_gate.tensor(error_gate2)
        elif error_type=="ampl":
            error=amplitude_damping_error(rate+fluc_rate*rnd.rand())
            if qn==2:
                error_gate=amplitude_damping_error(rate+fluc_rate*rnd.rand())
                error=error_gate.tensor(error_gate)
            elif qn==3:
                error_gate=amplitude_damping_error(rate+fluc_rate*rnd.rand())
                error_gate2=error_gate.tensor(error_gate)
                error=error_gate.tensor(error_gate2)
        if i==0:
            errors=error
        else:
            errors=errors.compose(error)
    noise_model.add_all_qubit_quantum_error(errors,gate_dict[qn])
    return noise_model

def add_noise(noise_dict,fluc_rate):
    noise_model=NoiseModel()
    if "single" in noise_dict.keys():
        noise_model=gen_noise(noise_model,1,noise_dict["single"],fluc_rate,noise_dict["seed"])
    if "two" in noise_dict.keys():
        noise_model=gen_noise(noise_model,2,noise_dict["two"],fluc_rate,noise_dict["seed"])
    if "three" in noise_dict.keys():
        noise_model=gen_noise(noise_model,3,noise_dict["three"],fluc_rate,noise_dict["seed"])
    return noise_model

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

def compute_expectation(qn,popu,w):
    return probs_expectation(popu,get_Zobs(w,qn))

def compute_exp(qn,circ,obs:int,nshots,noise_dict,noise=True,single_qubit=False,pidx=[],compiled=False):
    if single_qubit:
        #qn=n*order+1
        meas_qc=circ.copy()
        meas_qc.barrier()
        meas_qc.measure(0,0)
    else:
        meas_qc,w=add_obs_basis(circ,qn,obs)
        meas_qc.barrier()
        meas_qc.measure(list(range(qn)),list(range(qn)))
    backend=Aer.get_backend('aer_simulator')
    #add step:close automatic optimization
    if noise:
        if "fluctuation" in noise_dict.keys():
            fluc_rate=noise_dict["fluctuation"]
        else:
            fluc_rate=0
        noise_model=add_noise(noise_dict,fluc_rate)
        simulator=Aer.get_backend('qasm_simulator')
        if compiled:
            meas_qc=transpile(meas_qc,basis_gates=["cz","rx","ry","rz","h"],backend=simulator,optimization_level=3)
        result=simulator.run(meas_qc,noise_model=noise_model,shots=nshots).result()
    else:
        if compiled:
            meas_qc=transpile(meas_qc,basis_gates=["cz","rx","ry","rz","h"],backend=backend,optimization_level=3)
        job=backend.run(meas_qc,shots=nshots)
        result=job.result()
    counts=result.get_counts()
    zero_count=0
    if single_qubit:
        for k,v in counts.items():
            if k[-1]=="0":
                zero_count+=int(v)
        p0=float(zero_count/nshots)
        if obs>0:
            pid=pidx[2]
        else:
            pid=pidx[0]
        if pid>0:
            ps=int2string(pid,4,qn)
            if ps[0]=="3" or ps[0]=="2":
                p0=1-p0
        popu=np.array([p0,1-p0])
        # print(obs,popu)
        T,op=query_T(1)
        Ep=T.dot(np.real(popu))
        popu=op(Ep)
        expectation=2*popu[0]-1
    else:
        popu=np.zeros(2**qn,dtype=float)
        for outcome,count in counts.items():
            reversed_outcome=int(''.join(reversed(outcome)),2)
            popu[reversed_outcome]=float(count/nshots)
        popu=np.array(popu)/np.sum(popu)
        T,op=query_T(qn)
        Ep=T.dot(np.real(popu))
        popu=op(Ep)
        expectation=compute_expectation(qn,popu,w)

    return expectation

#--- about twirling
def complete_pidx(w,n,target_n):
    string=int2string(w,4,n)
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
    else:
        raise Exception()

def converge_pidx(n,pidx1,pidx2):
    ps1=int2string(pidx1,4,n)
    ps2=int2string(pidx2,4,n)
    new_ps=""
    for i in range(n):
        new_ps+=str(pauli_product(ps1[i],ps2[i]))
    return int(new_ps,base=4)   

def index2ps(qn,pidx):
    #pidx 2 cirq.paulistring
    q=cirq.LineQubit.range(qn)
    ps=int2string(pidx,4,qn)
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
    P1=index2ps(qn+1,int(int2string(pidx,4,order*qn+1)[:qn+1],base=4))
    P2=P1.conjugated_by(conjugate_circuit)
    return complete_pidx(ps2index(qn+1,P2),qn+1,order*qn+1)

def random_cswap_pidx(qn,idx,order):
    ps=["{}".format(np.random.choice([0,3],1)[0])]+["0"]*(order*qn)
    chosen="{}".format(np.random.choice([0,1,2,3],1)[0])
    for order_idx in range(order):
        ps[order_idx*qn+idx+1]=chosen
    # print(ps)
    return int("".join(ps),base=4)

def add_controlled_gate_cirq(circ,qubits,idx):
    if idx==1:
        circ.append(cirq.Moment(cirq.CX(*qubits)))
    elif idx==2:
        circ.append((cirq.S**(-1))(qubits[1]))
        circ.append(cirq.CX(*qubits))
        circ.append(cirq.S(qubits[1]))
    elif idx==3:
        circ.append(cirq.Moment(cirq.CZ(*qubits)))
    return circ 

def create_vd_obs_circ(qn,obs:int):
    q=cirq.LineQubit.range(qn+1)
    obs_string=int2string(obs,4,qn)
    circ=cirq.Circuit()
    for i,s in enumerate(obs_string):
        circ=add_controlled_gate_cirq(circ,[q[0],q[i+1]],int(s,base=4))
    return circ

def complete_cz_pidx(a,qn,order):
    a_str=int2string(a,4,2)
    obs=int(a_str[0]+"0"*(qn-1)+a_str[1],base=4)
    return complete_pidx(obs,qn+1,order*qn+1)

def compute_cz_conjugate(qn,cz_pidx,order):
    cz_pidx_string=int2string(cz_pidx,4,qn*order+1)
    cz_str=cz_pidx_string[0]+cz_pidx_string[qn]
    cz_back_str=cz_map[int(cz_str,base=4)]
    return complete_cz_pidx(int(cz_back_str,base=4),qn,order)

def random_list_vd(qn,order,random_instance,obs:int):
    obs_circ=create_vd_obs_circ(qn,obs)
    cswap_pidx_list=[]
    for i in range(random_instance*5):
        for ii in range(qn):
            if ii==0:
                ps=random_cswap_pidx(qn,ii,order)
            else:
                ps=converge_pidx(order*qn+1,ps,random_cswap_pidx(qn,ii,order))
        cswap_pidx_list.append(ps)
    cz_pidx_list=[complete_cz_pidx(a,qn,order) for a in np.random.randint(0,4**2,random_instance*5)]
    pidx_list=[]
    for i in range(random_instance*5):
        inter_idx=converge_pidx(order*qn+1,cswap_pidx_list[i], cz_pidx_list[i])
        cz_back_idx=compute_cz_conjugate(qn,cz_pidx_list[i],order)
        pidx_list.append([cswap_pidx_list[i],inter_idx,cz_back_idx])
    pidx_list=list(np.unique(pidx_list,axis=0))
    np.random.shuffle(pidx_list)
    pidx_list=pidx_list[:random_instance]
    return pidx_list

#--- SD
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

def compute_one_basis_shadow(qn,Ns,popu,basis):
    popu=np.array(popu)
    popu/=popu.sum()
    chosen_bit=np.random.choice(range(2**qn),size=Ns,p=popu)
    basis_rho=np.zeros((2**qn,2**qn),dtype=np.complex128)
    clifford_outcome_dict={"0":clifford_zero_list,"1":clifford_one_list}
    # print(includsing_index[0])
    for idx in chosen_bit:
        outcome_rho=continuous_tensor([clifford_outcome_dict[s][basis[i]-1] for i,s in enumerate(int2string(idx,2,qn))])
        # print(np.real(np.trace(outcome_rho)))
        basis_rho+=np.array(outcome_rho,dtype=np.complex128)
    return basis_rho/Ns

def compute_basis_list_shadow(qn,basis_list,popu_list,Ns):
    Nu=len(basis_list)
    shadow_list=[compute_one_basis_shadow(qn,Ns,popu_list[i],basis_list[i]) for i in range(Nu)]
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

def add_SD_gate(qc,idx,basis:int):
    if basis==0:
        pass
    elif basis==1:
        qc.x(idx)
    elif basis==2:
        qc.y(idx)
    elif basis==3:
        qc.z(idx)
    elif basis==4:
        qc.rx(np.pi/2,idx)
    elif basis==5:
        qc.ry(np.pi/2,idx)
    elif basis==6:
        qc.rx(-np.pi/2,idx)
    elif basis==7:
        qc.ry(-np.pi/2,idx)
    elif basis==8:
        qc.ry(np.pi/2,idx)
        qc.rx(np.pi/2,idx)
    elif basis==9:
        qc.ry(-np.pi/2,idx)
        qc.rx(np.pi/2,idx)
    elif basis==10:
        qc.ry(np.pi/2,idx)
        qc.rx(-np.pi/2,idx)
    elif basis==11:
        qc.ry(-np.pi/2,idx)
        qc.rx(-np.pi/2,idx)
    elif basis==12:
        qc.rx(np.pi/2,idx)
        qc.ry(np.pi/2,idx)
    elif basis==13:
        qc.rx(-np.pi/2,idx)
        qc.ry(np.pi/2,idx)
    elif basis==14:
        qc.rx(np.pi/2,idx)
        qc.ry(-np.pi/2,idx)
    elif basis==15:
        qc.rx(-np.pi/2,idx)
        qc.ry(-np.pi/2,idx)
    elif basis==16:
        qc.ry(np.pi/2,idx)
        qc.x(idx)
    elif basis==17:
        qc.ry(-np.pi/2,idx)
        qc.x(idx)
    elif basis==18:
        qc.rx(np.pi/2,idx)
        qc.y(idx)
    elif basis==19:
        qc.rx(-np.pi/2,idx)
        qc.y(idx)
    elif basis==20:
        qc.rx(np.pi/2,idx)
        qc.ry(np.pi/2,idx)
        qc.rx(np.pi/2,idx)
    elif basis==21:
        qc.rx(-np.pi/2,idx)
        qc.ry(np.pi/2,idx)
        qc.rx(-np.pi/2,idx)
    elif basis==22:
        qc.rx(np.pi/2,idx)
        qc.ry(np.pi/2,idx)
        qc.rx(-np.pi/2,idx)
    elif basis==23:
        qc.rx(np.pi/2,idx)
        qc.ry(-np.pi/2,idx)
        qc.rx(-np.pi/2,idx)
    return qc

def shadow_estimation(qn,qc,basis:list,noise_dict):
    #add SD gate
    circ=deepcopy(qc)
    for i in range(qn):
        circ=add_SD_gate(circ,i,basis[i]-1)
    circ.barrier()
    circ.measure(list(range(qn)),list(range(qn)))
    if "fluctuation" in noise_dict.keys():
        fluc_rate=noise_dict["fluctuation"]
    else:
        fluc_rate=0
    noise_model=add_noise(noise_dict,fluc_rate)
    simulator=Aer.get_backend('qasm_simulator')
    result=simulator.run(circ,noise_model=noise_model,shots=1000).result()
    counts=result.get_counts()
    popu=np.zeros(2**qn,dtype=float)
    for outcome,count in counts.items():
        reversed_outcome=int(''.join(reversed(outcome)),2)
        popu[reversed_outcome]=count/1000
    #noise process
    Ep=popu/sum(popu)
    T,op=query_T(qn)
    Ep=T.dot(np.real(popu))
    Ep=op(Ep)
    return Ep

def dachai_magic_function(n,O,rho):
    lam,lvec=np.linalg.eigh(rho)
    out=0
    for i in range(2**n):
        vechere=np.matrix(lvec[:,i]/np.linalg.norm(lvec[:,i]))
        out+=lam[i]**2*np.conjugate(vechere) @ O @ vechere.T
    return np.real(out[0,0])

def SD_expectation(qn,shadow_list,obs):
    Nu=len(shadow_list)
    ob_op=get_Pobs(obs,qn)
    OO=np.array(ob_op,dtype="complex128")
    rho_total=deepcopy(shadow_list[0])
    for j in range(1,Nu):
        rho_total+=shadow_list[j]
    out=dachai_magic_function(qn,OO,rho_total)
    for i in range(Nu):
        out-=dachai_magic_function(qn,OO,shadow_list[i])
    return out
#--- VD-based 
def generate_Pauli_eigenvec(qn,obs:int):
    obs_string=int2string(obs,4,qn)
    return continuous_tensor([eigen_ops[int(s,base=4)] for s in obs_string])

def generate_Pauli_eigencirc(qn,obs:int):
    obs_string=int2string(obs,4,qn)
    qc=QuantumCircuit(qn,qn)
    for i, s in enumerate(obs_string):
        if s=="1":
            qc.h(i)
        elif s=="2":
            qc.sdg(i)
            qc.h(i)
    return qc

def add_pauli(qc,n,pidx,first_qubit=False):
    if pidx>0:
        ps=int2string(pidx,4,n)
        for i,p in enumerate(ps):
            qc=add_single_pauli(qc,i,p)
            if first_qubit:
                break
    return qc

def add_controlled_gate(qc,qubits,s):
    if s=="X":
        qc.cx(*qubits)
    elif s=="Y":
        qc.cy(*qubits)
    elif s=="Z":
        qc.cz(*qubits)
    return qc

def toVD_circ(qn,order,circ):
    num_total_qubits=qn*order+1
    new_circ=QuantumCircuit(num_total_qubits,num_total_qubits)
    for idx in range(order):
        start_idx=idx*qn+1
        qubit_map={start_idx+qubit:qubit for qubit in range(qn)}
        new_circ.compose(circ,qubits=qubit_map,inplace=True)
    return new_circ

def VD_circuit(qn,obs:int,order,pidx_set,times):
    qc=QuantumCircuit(order*qn+1,order*qn+1)
    qc.h(0)
    qc=add_pauli(qc,order*qn+1,pidx_set[0])  
    for order_idx in range(order-1):
        for i in range(1,qn+1):
            for _ in range(times):
                qc.cswap(0,i+order_idx*qn,i+(order_idx+1)*qn)
    if obs>0:
        qc=add_pauli(qc,order*qn+1,pidx_set[1])
        pauli_string=int2Pauli_string(obs,4,qn)
        for i,s in enumerate(pauli_string):
            for _ in range(times):
                if s!="0":
                    qc=add_controlled_gate(qc,[0,i+1],s)
    qc.h(0)
    return qc

def query_VD_value(qn,obs:int,state_circuit,order,nshots,random_instance:int,noise_dict,initial_state=None,amplified_list=[1],compiled=False):
    if random_instance:
        pidx_list=random_list_vd(qn,order,random_instance,obs)
    else:
        pidx_list=[[0,0,0]]
    exp_noisy_list=[]
    for amplified in amplified_list:
        exp_list=[[],[]]
        for i,ob in enumerate([obs,0]):
            for pidx in pidx_list:
                vd_circuit=VD_circuit(qn,ob,order,pidx,amplified)
                if initial_state:
                    qc=QuantumCircuit(vd_circuit.num_qubits)
                    qc.initialize(initial_state,range(vd_circuit.num_qubits))
                    qc.compose(vd_circuit,qubits=range(qn*order+1),inplace=True)
                else:
                    qc=toVD_circ(qn,order,state_circuit)
                    qc.compose(vd_circuit,qubits=range(qn*order+1),inplace=True)
                exp=compute_exp(qn*order+1,qc,ob,nshots,noise_dict,single_qubit=True,pidx=pidx,compiled=compiled)
                exp_list[i].append(exp)
        exp_noisy_list.append([np.mean(exp_list[0]),np.mean(exp_list[1])])
    return exp_noisy_list

def noisyVD(qn,obs,state_circuit,order,nshots,noise_dict,initial_state=None,compiled=False):
    noisy_rho_exp=query_VD_value(qn,obs,state_circuit,order,nshots,0,noise_dict,initial_state=initial_state,compiled=compiled)[0]
    return noisy_rho_exp[0]/noisy_rho_exp[1]

def CNR_VD_cali(qn,obs,order,nshots,random_instance,noise_dict,compiled=False):
    ini_circuit=generate_Pauli_eigencirc(qn,obs)
    cali_values=query_VD_value(qn,obs,ini_circuit,order,nshots,random_instance,noise_dict,compiled=compiled)[0]
    cali_value=cali_values[0]/cali_values[1]
    return cali_value

def CNR_VD(qn,obs,state_circuit,order,nshots,random_instance,noise_dict,cali_value=False,initial_state=None,compiled=False):
    noisy_rho_exp=query_VD_value(qn,obs,state_circuit,order,nshots,random_instance,noise_dict,initial_state=initial_state)[0]
    noisy_exp=noisy_rho_exp[0]/noisy_rho_exp[1]
    if not cali_value:
        ini_circuit=generate_Pauli_eigencirc(qn,obs)
        cali_values=query_VD_value(qn,obs,ini_circuit,order,nshots,random_instance,noise_dict,compiled=compiled)[0]
        cali_value=cali_values[0]/cali_values[1]
    # print(noisy_exp,cali_value)
    return noisy_exp/cali_value

def ZNE_VD(qn,obs,state_circuit,order,nshots,random_instance,noise_dict,initial_state=None,compiled=False):
    amplified_list=[1,3]
    noisy_rho_exp_list=query_VD_value(qn,obs,state_circuit,order,nshots,random_instance,noise_dict,amplified_list=amplified_list,initial_state=initial_state,compiled=compiled)
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

def SD(qn,obs,Nu,Ns,state_circ,noise_dict):
    exp_list=[]
    for ob in [obs,0]:
        basis_list=[[np.random.randint(1,25) for _ in range(qn)] for _ in range(Nu)]
        popu_list=[shadow_estimation(qn,state_circ,basis,noise_dict) for basis in basis_list]
        shadow_list=compute_basis_list_shadow(qn,basis_list,popu_list,Ns)
        # print(shadow_list[0])
        exp_list.append(SD_expectation(qn,shadow_list,ob))
    # print(exp_list)
    return exp_list[0]/exp_list[1]

#--- control group 
def Unmit(qn,obs,state_circuit,nshots,noise_dict,noise=True):
    qc=deepcopy(state_circuit)
    return compute_exp(qn,qc,obs,nshots,noise_dict,noise=noise)

def idealVD(qn,obs,initial_state,order):
    distill_state=continuous_matmul([initial_state]*order)
    obs_op=get_Pobs(obs,qn)
    nomi=np.real(np.trace(distill_state@obs_op))
    denomi=np.real(np.trace(distill_state))
    return nomi/denomi

#--- Experimental settings

#random state
def orthogonalize(vector,basis):
    random_orthogonal_vector=vector-np.dot(vector, np.conj(basis))/np.dot(basis,np.conj(basis))*basis
    random_orthogonal_vector/=np.linalg.norm(random_orthogonal_vector)
    return random_orthogonal_vector

def generate_orthogonal_part(qn,seed):
    dominant=rand_vec(qn,seed)
    errornous=orthogonalize(rand_vec(qn,seed+1000),dominant)
    return vec2rho(dominant),vec2rho(errornous)

def random_state_purity(qn,p0,seed):
    p1=1-p0
    dominant,errornous=generate_orthogonal_part(qn,seed)
    return p0*dominant+p1*errornous,dominant

#specific circuit
def add_TS_block(circ,q1,q2,theta):
    circ.cx(q1,q2)
    circ.rz(theta,q2)
    circ.cx(q1,q2)
    return circ

def TS_circ(qn,step,delta,ratio):
    circ=QuantumCircuit(qn,qn)
    for _ in range(step):
        for i in range(qn):
            circ.rx(delta,i)
        for i in range(int(qn/2)):
            circ=add_TS_block(circ,2*i,2*i+1,-delta*ratio)
        for i in range(int((qn-1)/2)):
            circ=add_TS_block(circ,2*i+1,2*i+2,-delta*ratio)
    return circ

def fig4_data(qn,step,nshots,noise_dict,Nu,Ns,state_seed,random_instance,filename,cali_value=False):
    data_save_file=datapath+"/{}".format(filename)
    rnd=np.random.RandomState(state_seed)
    delta_range=[0.05,0.2]
    ratio_range=[0.2,1.5]
    delta=rnd.uniform(delta_range[0],delta_range[1])
    ratio=rnd.uniform(ratio_range[0],ratio_range[1])
    state_circ=TS_circ(qn,step,delta,ratio)
    obs=int("0"*(qn-1)+"3",base=4)
    unmit=Unmit(qn,obs,deepcopy(state_circ),nshots,noise_dict)
    print("unmit",unmit)
    ideal=Unmit(qn,obs,deepcopy(state_circ),int(1e7),noise_dict,noise=False)
    print("ideal",ideal)
    noisy_vd=noisyVD(qn,obs,deepcopy(state_circ),2,nshots,noise_dict)
    print("noisy_vd",noisy_vd)
    cnr_vd=CNR_VD(qn,obs,deepcopy(state_circ),2,nshots,random_instance,noise_dict,cali_value=cali_value)
    print("cnr_vd",cnr_vd)
    zne_vd=ZNE_VD(qn,obs,deepcopy(state_circ),2,int(nshots/2),random_instance,noise_dict)
    noisy_vd_no=noisyVD(qn,obs,deepcopy(state_circ),2,nshots,noise_dict,compiled=True)
    print("noisy_vd",noisy_vd)
    cnr_vd_no=CNR_VD(qn,obs,deepcopy(state_circ),2,nshots,random_instance,noise_dict,cali_value=cali_value,compiled=True)
    print("cnr_vd",cnr_vd)
    zne_vd_no=ZNE_VD(qn,obs,deepcopy(state_circ),2,int(nshots/2),random_instance,noise_dict,compiled=True)
    print("zne_vd",zne_vd)
    sd=SD(qn,obs,Nu,Ns,state_circ,noise_dict)
    print("sd",sd)
    data=pd.DataFrame({"qn":[str(qn)],"seed":[str(state_seed)],"obs":[str(obs)],"nshots":[str(nshots)],"step":[str(step)],"random_instance":[str(random_instance)],"ideal":[str(ideal)],"sd":[str(sd)],"noisy_vd":[str(noisy_vd)],"cnr_vd":[str(cnr_vd)],"zne_vd":[str(zne_vd)],"noisy_vd_no":[str(noisy_vd_no)],"cnr_vd_no":[str(cnr_vd_no)],"zne_vd_no":[str(zne_vd_no)]})
    savedata(data,data_save_file,cover_old=False)

def fig4(qn,step,nshots,noise_dict,Nu,Ns,state_seed_list,random_instance):
    filename="TS_no_compile.csv"
    obs=int("0"*(qn-1)+"3",base=4)
    noise_dict_=deepcopy(noise_dict)
    noise_dict_["seed"]=0
    cali_value=CNR_VD_cali(qn,obs,2,int(1e5),random_instance,noise_dict_)
    for seed in state_seed_list:
        noise_dict_=deepcopy(noise_dict)
        noise_dict_["seed"]=seed
        fig4_data(qn,step,nshots,noise_dict_,Nu,Ns,seed,random_instance,filename,cali_value=cali_value)


#--- test
if __name__=="__main__":
    noise_dict={"fluctuation":1e-5,"single":[["depo",sy23_1e],["ampl",sy23_decay]],"two":[["depo",sy23_2e],["ampl",sy23_decay]],"three":[["depo",pd3],["ampl",sy23_decay]]}
    random_instance=4
    nshots_list=[1e3,5e3,1e4,2e4,5e4,8e4,1e5]
    step_list=[2,4,6]
    state_seed_list=list(range(20))
    qn_list=[2,4,6,8]
    for qn in qn_list:
        for step in step_list:
            for nshots in nshots_list:
                if nshots_list>1e4:
                    Ns=50
                else:
                    Ns=10
                Nu=int(nshots/Ns)
                fig4(qn,step,nshots,noise_dict,Nu,Ns,state_seed_list,random_instance)