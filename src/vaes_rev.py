
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import csr_matrix

import anndata
import scvi
import scanpy as sc

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

from pyro.ops.indexing import Vindex
from pyro import poutine
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, TraceMeanField_ELBO



class NAPSets(nn.Module):
    def __init__(self,
                 n_genes,
                 n_channels,
                 probes_per_gene= 20,
                 total_probes= 5e4):
        super().__init__()
        self.mxpr= total_probes
        self.cnst= probes_per_gene

        self.gns= nn.Parameter(torch.zeros(n_genes, 1))
        self.prp= nn.Parameter(torch.zeros(n_genes, n_channels))
        self.ttl= nn.Parameter(torch.tensor(3.))

    def get_probes(self):
        prs= self.prp.softmax(1) * (self.gns.softmax(0) * self.mxpr * self.ttl.sigmoid()).clamp(1e-3, self.cnst)
        prx= prs + (prs.round() - prs).detach()
        return prx

    def forward(self, X, wprx= False):
        prx= self.get_probes()
        if not wprx:
            return X.mm(prx) +1
        else:
            return X.mm(prx) +1, prx


class VAEBase(nn.Module):
    def __init__(self, adata, n_prj=24, probes_per_gene=100, total_probes=5e4, lblwt= .5):
        super().__init__()
        self.adata= adata
        self.genes= adata.var_names.tolist()
        self.n_cll= adata.obs._scvi_labels.unique().size
        self.n_src= adata.obs._scvi_batch.unique().size
        self.n_gns= adata.shape[1]
        self.n_prj= n_prj
        self.lblwt= lblwt
        self.dat_samp= None

        self.dsc= {
            'n_prj': self.n_prj,
            'n_cll': self.n_cll,
            'n_src': self.n_src,
            'n_gns': self.n_gns,
        }

    def dat_iter(self, n_iter, n_btch= 2, lbls= True):
        index= {k:v.index for k,v in self.adata.obs.groupby(['_scvi_batch','_scvi_labels'])}
        if isinstance(self.adata.X, csr_matrix): self.adata.X= self.adata.X.toarray()

        for i in range(n_iter):
            s,l,c=[],[],[]
            for k,v in index.items():
                inds= np.random.choice(v, n_btch, replace=True)
                c.append(torch.FloatTensor(self.adata[inds].X))
                l.append(torch.LongTensor([k[1]]*n_btch))
                s.append(torch.LongTensor([k[0]]*n_btch))
            yield {'cts':torch.cat(c),
                   'src':torch.cat(s),
                   'lbl':None if not lbls else torch.cat(l),
                   'prg':i/float(n_iter)}

    def train(self, n_iter= 1500, n_btch= 2, lr= 1e-2, update= False, verbose=True, lbls= True, loss= Trace_ELBO()):
        svi = SVI(self.model, self.guide,
                  ClippedAdam({'lr': lr}),
                  loss=loss)
        if not update: pyro.clear_param_store()

        for itr, dat in tqdm(enumerate(self.dat_iter(n_iter, n_btch, lbls)), total=n_iter):
            elbo= svi.step(dat)
            if not itr%(n_iter//10) and verbose:
                r2= np.corrcoef(self.get_means(dat).view(-1), dat['cts'].view(-1))[0,1]**2
                prc= 1 if not lbls else (self.get_assignment(dat)==dat['lbl']).float().mean()
                print('%d | elbo: %.2E | prc: %.2f | r2: %.2f'%(itr, elbo, prc, r2))

        self.dat_samp= dat


class VAEGamma(VAEBase):
    def __init__(self, adata, n_prj, probes_per_gene,
                 total_probes, lblwt, loc_m= None,
                 beta_m= None, beta_p= None):
        super().__init__(adata, n_prj, probes_per_gene, total_probes, lblwt)
        self.loc_m= loc_m
        self.beta_m= beta_m
        self.beta_p= beta_p
        self.znflgt= nn.Linear(self.n_prj, self.n_gns)
        self.enc= NAPSets(self.n_gns, self.n_prj, probes_per_gene= probes_per_gene, total_probes= total_probes)

    def model(self, data):

        if self.beta_m is not None:
            beta= (torch.ones(self.n_prj)*self.beta_m).log()
        else:
            beta= pyro.sample('beta', dist.Normal(0,1).expand((self.n_prj,)).to_event(1))

        if self.loc_m is not None:
            loc_prj= (torch.ones(self.n_cll,self.n_prj)*self.loc_m).log()
        else:
            loc_prj= pyro.sample('loc_prj', dist.Normal(0,1).expand((self.n_cll, self.n_prj)).to_event(2))

        pyro.sample('prx_reg', dist.Gamma(.3,.03).expand((self.n_gns, self.n_prj)).to_event(2))

        dir_prj= pyro.sample('dir_prj', dist.Gamma(1,1).expand((self.n_cll, self.n_prj)).to_event(2))
        decoder= pyro.param('decoder', lambda: torch.zeros(self.n_prj, self.n_gns)).softmax(1)
        pyro.module('znflgt', self.znflgt)
        with pyro.plate('obs', data['cts'].shape[0]):
            ctg= pyro.sample('ctg', dist.Categorical(logits=torch.zeros(self.n_cll)), obs= data['lbl'])
            prj= pyro.sample('prj', dist.Gamma(loc_prj[ctg].exp()*beta.exp(), beta.exp()).to_event(1))
            nrm= pyro.sample('nrm', dist.Dirichlet(dir_prj[ctg]))
            pyro.sample('obs_cts', dist.Poisson(nrm.mm(decoder)*data['cts'].sum(1).unsqueeze(1)).to_event(1),
                        obs= data['cts'])

    def guide(self, data):
        pars= {}
        pars['beta']= pyro.sample('beta', dist.Delta(pyro.param('beta_loc', lambda: torch.ones(self.n_prj,))).to_event(1))
        pars['loc_prj']= pyro.sample('loc_prj', dist.Delta(pyro.param('loc_prj_loc',
                                      lambda: torch.zeros(self.n_cll, self.n_prj))).to_event(2))

        pyro.module('enc', self.enc, update_module_params=data['lbl'] is not None)
        prj_beta_loc= self.beta_p if self.beta_p is not None else pyro.param('prj_beta_loc', lambda: torch.ones(1), constraint= constraints.positive)
        prj_mu_loc, prx_loc= self.enc(data['cts'], True)
        pyro.sample('prx_reg', dist.Delta(prx_loc +1e-6).to_event(2))

        dir_prj_loc= pyro.param('dir_prj_loc', lambda: torch.ones(self.n_cll, self.n_prj), constraint= constraints.interval(1e-2,1e2))
        dir_prj_beta= pyro.param('dir_prj_beta', lambda: torch.ones(1), constraint= constraints.positive)
        pars['dir_prj']= pyro.sample('dir_prj', dist.Gamma(dir_prj_loc *dir_prj_beta, dir_prj_beta).to_event(2))

        pars['dir_gns']= pyro.param('decoder', lambda: torch.zeros(self.n_prj, self.n_gns)).softmax(1)
        with pyro.plate('obs', data['cts'].shape[0]):
            pars['prj']= pyro.sample('prj', dist.Gamma(prj_mu_loc *prj_beta_loc, prj_beta_loc).to_event(1))
            pars['nrm']= pyro.sample('nrm', dist.Delta(pars['prj']/pars['prj'].sum(1).unsqueeze(-1)).to_event(1))
            pars['lgt']= dist.Dirichlet(pars['dir_prj']).log_prob(pars['nrm'].unsqueeze(1))
            ctg_dst= dist.Categorical(logits=pars['lgt'])

            if data['lbl'] is not None:
                pyro.factor('ctg_lss', -self.lblwt*ctg_dst.log_prob(data['lbl']))

            else:
                pars['ctg']= pyro.sample('ctg', ctg_dst)
        return pars

    def get_assignment(self, data):
        with torch.no_grad():
            pars= self.guide(data)
            return pars['lgt'].max(1)[1]

    def get_means(self, data):
        with torch.no_grad():
            pars= self.guide(data)
            return pars['nrm'].mm(pars['dir_gns']) *data['cts'].sum(-1).unsqueeze(-1)
