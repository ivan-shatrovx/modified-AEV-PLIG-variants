#!/usr/bin/env python
# coding: utf-8

# # Scripts for preparing metatdata for PDBBind-Opt
# 
# In this script, we will create CSV files containing relevant metadata that will be used for the PDBBind-Opt workflow to process structures.

# In[1]:


import pandas as pd
from tqdm import tqdm
import re
import requests
import math

from rdkit import Chem


# In[2]:


def get_smiles_from_rcsb(comp_id: str):
    """
    Query ligand SMILES from RCSB

    Parameters
    ----------
    comp_id: str
        The ligand ID, usually a three-letter code

    Returns
    -------
    smi: str
        The SMILES of the query ligand. If fail to get, will return a vacant string
    """
    query = '''{chem_comp(comp_id: "%s") {
        rcsb_chem_comp_descriptor {
        SMILES_stereo SMILES InChI
        }
    }
    }''' % comp_id
    query = re.sub(r'\s+', ' ', query)
    try:
        res = requests.get('https://data.rcsb.org/graphql?query=' + query)
        smi = res.json()['data']['chem_comp']['rcsb_chem_comp_descriptor']['SMILES_stereo']
        if smi is None:
            smi = res.json()['data']['chem_comp']['rcsb_chem_comp_descriptor']['SMILES']
        if smi is None:
            m = Chem.MolFromInchi(res.json()['data']['chem_comp']['rcsb_chem_comp_descriptor']['InChI'])
            smi = Chem.MolToSmiles(m)
        assert smi is not None, "No reference smiles"
        return smi
    except:
        return ""


def regularize_binding_data(typ, sign, number, unit):

    # handle number that have uncertainty
    if '+-' in number:
        number = number.split('+-')[0]
    number = float(number)
    # handle sign
    sign = sign[1] + sign[0] if sign in ['=>', '=<'] else sign
    # convert Ka/Kb to Kd
    typ = typ.lower()
    if typ == 'ka' or typ == 'kb':
        typ = 'kd'
        assert unit.endswith('^-1'), f'Incorrect unit for Ka/Kb: {unit}'
        unit = unit.rstrip('^-1')
        number = 1 / number

    if unit == 'M':
        lognum = math.log10(number)
    elif unit == 'mM':
        lognum = math.log10(number) - 3
    elif unit == 'uM':
        lognum = math.log10(number) - 6
    elif unit == 'nM':
        lognum = math.log10(number) - 9
    elif unit == 'pM':
        lognum = math.log10(number) - 12
    elif unit == 'fM':
        lognum = math.log10(number) - 15
    else:
        lognum = None

    return {
        "measurement": typ,
        "sign": sign,
        "value": number,
        "unit": unit,
        "logvalue": lognum
    }


# ## Parse Original PDBBind Data

# In[3]:


def parse_pdbbind_metadata(index='../index/INDEX_general_PL.2020R1.lst'):
    data = []
    with open(index) as f:
        for line in f:
            if line.startswith('#'):
                continue
            if line:
                content = line.strip().split()
                if not content[6].endswith(')'):
                    ligand = content[6][1:]
                else:
                    ligand = content[6][1:-1]

                data.append({
                    "PDBID": content[0],
                    "Resolution": content[1],
                    "Year": content[2],
                    "Binding Affinity": content[3],
                    "Ligand": ligand.lstrip('_'),
                    "Note": ' '.join(content[7:])
                })
    data = pd.DataFrame(data)
    return data

pdbbind_data = parse_pdbbind_metadata('../index/INDEX_general_PL.2020R1.lst').set_index("PDBID").sort_index()
pdbbind_ids = pdbbind_data.index.unique()
print("Number of data in PDBBind v2020:", len(pdbbind_ids))
pdbbind_data


# ## Parse BioLiP

# In[ ]:


#get_ipython().run_cell_magic('bash', '', 'wget https://zhanggroup.org/BioLiP/download/BioLiP.txt.gz\ngunzip BioLiP.txt.gz\n')


# In[4]:


columns = [
    'PDBID',
    'Receptor chain',
    'Resolution',
    'Binding site',
    'Ligand CCD',
    'Ligand chain',
    'Ligand serial number',
    'Binding site residues',
    'Binding site residues renumbered',
    'Catalytic site residues',
    'Catalytic site residues renumbered',
    'EC number',
    'GO terms',
    'Binding affinity (manual)',
    'Binding affinity (Binding MOAD)',
    'Binding affinity (PDBbind-CN)',
    'Binding affinity (Binding DB)',
    'UniProt ID',
    'PubMed ID',
    'Ligand residue sequence number',
    'Receptor sequence'
]
raw_df = pd.read_csv('BioLiP.txt', sep='\t', names=columns, low_memory=False, keep_default_na=False, na_values=[None, ""])
raw_df


# ## Prepare PDBBind-Opt

# In[18]:


biolip_pdbbind = raw_df.query('PDBID in @pdbbind_ids')
biolip_pdbbind_dict = {pdbid: subdf for pdbid, subdf in biolip_pdbbind.groupby("PDBID")}

datas = {
    "sm": [],
    "poly": []
}
patt = re.compile(r"([a-zA-Z50]+)([~<>=]+)([\d.eE+-]+)([^\s,]+)")
for pdbid, row in tqdm(list(pdbbind_data.iterrows())):
    # determine if small molecule or polymers
    ligand_ccd = row['Ligand']
    if bool(re.search(r'[^a-zA-Z0-9]', ligand_ccd)):
        category = 'poly'
    else:
        category = 'sm'


    # parse binding data
    binding_string = str(row['Binding Affinity'])
    binding_data = regularize_binding_data(*tuple(re.findall(patt, binding_string))[0])
    binding_data['source'] = 'PDBBind'
    binding_data['origin'] = binding_string

    if category == 'poly' or (pdbid not in biolip_pdbbind_dict):
        record = {"PDBID": pdbid, "Ligand CCD": ligand_ccd, 'Ligand chain': None, 'Ligand residue sequence number': None}
        record.update(binding_data)
        datas[category].append(record)
    else:
        biolip_record = biolip_pdbbind_dict[pdbid]
        cnt = 0
        for chain, subdf in biolip_record.groupby('Ligand chain'):
            for _, row in subdf.iterrows():
                if row['Ligand CCD'] == ligand_ccd:
                    record = {
                        "PDBID": pdbid, "Ligand CCD": ligand_ccd, 
                        'Ligand chain': row['Ligand chain'], 
                        'Ligand residue sequence number': row['Ligand residue sequence number'].replace(' ', '')
                    }
                    record.update(binding_data)
                    datas[category].append(record)
                    cnt += 1
                    break
        if cnt == 0:
            record = {"PDBID": pdbid, "Ligand CCD": ligand_ccd, 'Ligand chain': None, 'Ligand residue sequence number': None}
            record.update(binding_data)
            datas[category].append(record)



# In[19]:


for category in datas:
    datas[category] = pd.DataFrame(datas[category])
    print(f"PDBBind-{category} #PDBID: {datas[category]['PDBID'].unique().shape[0]}")
    datas[category].to_csv(f'PDBBind_{category}.csv', index=None)

