#!/usr/bin/env python
# coding: utf-8

# ##Exercice 2)

# :

# In[1]:

import warnings
warnings.filterwarnings("ignore")
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('mnist-original')
X = mnist.data.astype('float64')
y = mnist.target.astype('float64')
print(X)
print(mnist)



# In[3]:


X.data.shape


# In[4]:


y.data.shape


# In[5]:


np.unique(y)


# In[6]:


print(y)


# In[7]:


msk = [(chiffre==7)or(chiffre==3)  for chiffre in y]
#print(msk)
Xbis= X[msk]
ybis= y[msk]
print(ybis)
plt.imshow((Xbis[14433]).reshape(28,28))#size Xbis=14433


#  Utiliser la fonction LogisticRegression pour apprendre un modèle de classication sur 
# l'intégralité des données (on choisira un cas sans ordonnée à l'origine, i.e., l'option fit_intercept=False). 
# Le modèle prédit alors la classe d'une image en considérant une image comme un vecteur x et en choisissant l'une des deux classes selon le signe de wJx, où w est le vecteur appris par la méthode et stocké dans l'attribut .coef_. 

# In[8]:


import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
t0 = time.time()


# In[9]:


clf = LogisticRegression(random_state=0, solver='lbfgs',fit_intercept=False).fit(Xbis, ybis)
w=clf.coef_
len(w)


# In[10]:


def PredictNumber(x):
    "predict class x"
    a=np.dot(w,x)
    if a>0:
        print("le chiffre est 7")
    else:
        print("le chiffre est 3")
    #plt.mshow


# In[11]:


PredictNumber(Xbis[1])
PredictNumber(Xbis[-1])


# In[12]:


x=Xbis[-1]
def fig_digit(alpha):
  x_mod=x-alpha*(np.dot(w,x)/np.dot(w,w.T))*w
  plt.subplot(121)
 
  plt.title('image modifié ')
  plt.imshow(np.matrix(x_mod).reshape(28,28))#, cmap=plt.cm.gray_r, interpolation='nearest')
  plt.show() 


# In[13]:


fig_digit(2)

plt.subplot(122)
plt.title('image initiale')
plt.imshow(np.matrix(x).reshape(28,28))#, cmap=plt.cm.gray_r, interpolation='nearest')


# In[14]:


from ipywidgets import interact, interactive, IntSlider, Layout
import ipywidgets as widgets
from IPython.display import display


# In[15]:


y=interactive(fig_digit,x=(x),w=(w),alpha=(0.1,100,0.1))
display(y)

#pour comparer
plt.subplot(122)
plt.title('image initiale')
plt.imshow(np.matrix(x).reshape(28,28))#, cmap=plt.cm.gray_r, interpolation='nearest')


# In[43]:


from sklearn.decomposition import PCA

acp = PCA(2)  # project from 64 to 2 dimensions
Xacp = acp.fit_transform(Xbis)
print(Xbis.shape)
print(Xacp.shape)


# In[46]:



plt.scatter(Xacp[ybis==7, 0], Xacp[ybis==7, 1],
            color='red',lw=2,label='chiffre 7')
plt.scatter(Xacp[ybis==3, 0], Xacp[ybis==3, 1],
            lw=2,label='chiffre 3')
plt.title('ACP du jeu donnée MNIST en 2 dimension')
plt.legend(loc='best',shadow=False,scatterpoints=1)
plt.figure()


# In[ ]:


from matplotlib import animation

get_ipython().run_line_magic('matplotlib', 'notebook')
fig = plt.figure()
ax = plt.axes(xlim=(0,28),ylim=(28,0))
ims = []

for i in range(100):
    im = fig_digit(i)
    ims.append([im])
#X[0],clf.coef_,
#anim = animation.FuncAnimation(fig, ims,frames=200,frames=200, interval=50, blit=True)
anim = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=50)
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
anim.to_html5_video()
plt.show()



