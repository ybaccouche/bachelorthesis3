{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import wget, os, gzip, pickle, random, re, sys\n",
    "import torch\n",
    "import torchvision\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import DataLoader, TensorDataset\n",
    "import math"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'Perceiver' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m/var/folders/yc/d0b45_gs319592768hmxskbm0000gn/T/ipykernel_53527/3611345343.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;31m# Parameters and setup\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m \u001b[0mmodel\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mPerceiver\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mk\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m256\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mheads\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdepth\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m3\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mseq_length\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m256\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnum_tokens\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m256\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnum_classes\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m256\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      5\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mto\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0md\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0mcriterion\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mNLLLoss\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'Perceiver' is not defined"
     ]
    }
   ],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
