#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import os
import sys
import subprocess
import string
import time
import signal
from threading import Thread
import datetime
import numpy as np
import random
import math
import logging

## PROCESSING
from astropy.io import ascii
from astropy.table import Table
import pandas as pd

## DRAW
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.rcParams['savefig.dpi'] = 300
#matplotlib.rcParams["figure.dpi"] = 300
import seaborn as sns

##################################################
###          MAIN
##################################################

# - Get args
inputfile= sys.argv[1]
classid_sel= int(sys.argv[2])
outfile= "plot.pdf"
if len(sys.argv)>3 and sys.argv[3]!="":
	outfile= sys.argv[3]

# - Read ascii 
table= ascii.read(inputfile)
colnames= table.colnames

print("colnames")
print(colnames)
print("dtype")
print(table.dtype)
print(type(table.dtype))

dtypes= [table.dtype[i] for i in range(len(table.colnames))]

print("dtypes")
print(dtypes)

# - Select class id
if classid_sel!=-1:
	table_sel= Table(names=colnames, dtype=dtypes)
	##table_sel= Table(names=table.colnames)

	for item in table:
		class_id= item["id"]
		if class_id!=classid_sel:
			continue
		table_sel.add_row(item)

else:
	table_sel= table

# - Convert to pandas
print("Convert to pandas ...")
df = table_sel.to_pandas()
print(df.head())

# - Remove sname & class id columns
print("Remove sname & class id from columns ...")
X= df.drop(labels=["sname", "id"], axis=1)   
print(X.head())

# - Compute correlation coefficient
cor= X.corr()
print("--> correlation matrix")
print(cor)

# - Generate tringular matrix mask
mask= np.ones_like(cor, dtype=bool)
#mask = np.triu(np.ones_like(cor, dtype=bool))
#mask = np.tril(np.ones_like(cor, dtype=bool))
mask_triu = np.triu(np.ones_like(cor, dtype=bool))
mask_tril = np.tril(np.ones_like(cor, dtype=bool))
#mask[mask_triu==True]= False
mask[mask_tril==True]= False

# - Draw
plt.figure(figsize=(12,10))

#ax= sns.heatmap(cor, linewidth=.5, annot=True, fmt=".1f", cmap="coolwarm", cbar_kws={'label': 'Pearson Correlation Coefficient'})
#ax= sns.heatmap(cor, mask=mask, vmin=-1, vmax=1, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"fontsize":10}, cbar_kws={'label': 'Pearson Correlation Coefficient', "orientation": "horizontal"})
#ax= sns.heatmap(cor, mask=mask, vmin=-1, vmax=1, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"fontsize":10}, cbar_kws={"use_gridspec": False, "location": "top", 'label': 'Pearson Correlation Coefficient'})
ax= sns.heatmap(cor, mask=mask, vmin=-1, vmax=1, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"fontsize":14}, cbar=False)

#ax.xaxis.tick_top() # x labels on top
plt.xticks(rotation=-45) # rotate x labels
xaxis_labels= [item.get_text()  for item in ax.get_xticklabels()]
pos, textvals = plt.xticks()
plt.xticks(pos + 0.5, xaxis_labels) # shift x ticks/labels by 0.5
plt.tick_params(left=False, bottom=False, top=False)  # hide ticks
#cbar = ax.collections[0].colorbar
#cbar.ax.tick_params(labelsize=15)
#ax.figure.axes[-1].yaxis.label.set_size(20)
#ax.figure.axes[-1].xaxis.label.set_size(20)

ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 14)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 14)

plt.tight_layout()
#plt.show()

plt.savefig(outfile)

