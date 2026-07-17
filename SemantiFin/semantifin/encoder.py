from __future__ import annotations
import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class StructuredEncoder:
    def __init__(self,cfg):
        self.cfg=cfg; self.scaler=StandardScaler(); self.pca=None; self.angle_scaler=MinMaxScaler((0,1))
    def _build256(self,row):
        blocks=[]
        # 64 event/claim block
        b1=np.zeros(64); e=np.asarray(row.event_distribution,float); p=np.asarray(row.polarity_distribution,float)
        b1[:len(e)]=e; b1[16:16+len(p)]=p; b1[24]=row.count; b1[25]=row.drift
        # 64 narrative block from fused embedding projection by deterministic pooling
        emb=np.asarray(row.fused_embedding,float); b2=np.array([emb[i::64].mean() if len(emb[i::64]) else 0 for i in range(64)])
        # 64 risk block
        b3=np.zeros(64); risk=np.asarray(row.risk_vector,float); b3[:len(risk)]=risk; b3[16]=risk.mean(); b3[17]=risk.max(); b3[18]=-(risk[risk>0]*np.log(risk[risk>0]+1e-12)).sum()
        # 64 credibility/provenance block
        b4=np.zeros(64); b4[0]=row.mean_credibility; b4[1]=row.source_diversity; b4[2]=row.count; b4[3]=row.drift
        return np.concatenate([b1,b2,b3,b4])
    def fit_transform(self,df:pd.DataFrame):
        X=np.stack([self._build256(r) for r in df.itertuples(index=False)])
        Xs=self.scaler.fit_transform(X)
        n=min(self.cfg.get("reduced_dimension",128),Xs.shape[0],Xs.shape[1])
        self.pca=PCA(n_components=n,random_state=42); Z=self.pca.fit_transform(Xs)
        target=self.cfg.get("reduced_dimension",128)
        if n<target: Z=np.pad(Z,((0,0),(0,target-n)))
        A=np.pi*self.angle_scaler.fit_transform(Z)
        norms=np.linalg.norm(Z,axis=1,keepdims=True); AMP=Z/np.where(norms==0,1,norms)
        Xhat=self.scaler.inverse_transform(self.pca.inverse_transform(Z[:,:n]))
        fidelity=1-np.linalg.norm(X-Xhat)/(np.linalg.norm(X)+1e-12)
        return X,Z,A,AMP,float(fidelity)
    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True); joblib.dump(self,path)
