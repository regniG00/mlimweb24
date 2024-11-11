import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap as up
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.datasets import fetch_covtype, fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, trustworthiness
from sklearn.utils import resample
#Preparation
##Fashion-MNIST
fdata = fetch_openml(name = 'fashion-mnist')
print("Download of Fashion-MNIST successful")
ftargetFrame = fdata['target'].to_frame().T
fdf = pd.DataFrame(fdata['data'], columns=fdata['feature_names'])
fscaler = StandardScaler()
fscaler.fit(fdf)
fscaled_data = fscaler.transform(fdf)

##Satellite
sdata = fetch_openml(name = 'one-hundred-plants-margin', version=1)
print("Download of one-hundred-plants-margin Dataset successful")
stargetFrame = sdata['target'].to_frame().T
sdf = pd.DataFrame(sdata['data'], columns=sdata['feature_names'])
sscaler = StandardScaler()
sscaler.fit(sdf)
sscaled_data = sscaler.transform(sdf)

##Credit-g
cdata = fetch_openml('isolet', version=1)
print("Download of isolet Dataset successful")
ctargetFrame = cdata['target'].to_frame().T
cdf = pd.DataFrame(cdata['data'], columns=cdata['feature_names'])
cscaler = StandardScaler()
cscaler.fit(cdf)
cscaled_data = cscaler.transform(cdf)


#Begin of training Phase
print("Starting PCA training")
fpca = PCA(n_components=2)
fpca.fit(fscaled_data)
pcad_fdata = fpca.transform(fscaled_data)

spca = PCA(n_components=2)
spca.fit(sscaled_data)
pcad_sdata = spca.transform(sscaled_data)

cpca = PCA(n_components=2)
cpca.fit(cscaled_data)
pcad_cdata = cpca.transform(cscaled_data)
print("PCA training finished")

print("Starting ICA training")
fica = FastICA(n_components=2)
fica.fit(fscaled_data)
icad_fdata = fica.transform(fscaled_data)

sica = FastICA(n_components=2)
sica.fit(sscaled_data)
icad_sdata = sica.transform(sscaled_data)

cica = FastICA(n_components=2)
cica.fit(cscaled_data)
icad_cdata = cica.transform(cscaled_data)
print("ICA training finished")

print("Starting LDA training")
flda = LinearDiscriminantAnalysis()
flda.fit(fscaled_data, fdata['target'])
lda_fdata = flda.transform(fscaled_data)

slda = LinearDiscriminantAnalysis()
slda.fit(sscaled_data, sdata['target'])
lda_sdata = slda.transform(sscaled_data)

clda = LinearDiscriminantAnalysis( )
clda.fit(cscaled_data, cdata['target'])
lda_cdata = clda.transform(cscaled_data)
print("LDA training finished")

print("Starting UMAP training, this may take a while")
freducer = up.UMAP(n_components=2)
ump_fdata = freducer.fit_transform(fscaled_data)

sreducer = up.UMAP(n_components=2)
ump_sdata = sreducer.fit_transform(sscaled_data)

creducer = up.UMAP(n_components=2)
ump_cdata = creducer.fit_transform(cscaled_data)
print("UMAP training finished")

print("Starting T-SNE training, this may take a while")
ftsn = TSNE(n_components=2)
tsn_fdata = ftsn.fit_transform(fscaled_data)

stsn = TSNE(n_components=2)
tsn_sdata = stsn.fit_transform(sscaled_data)

ctsn = TSNE(n_components=2)
tsn_cdata = ctsn.fit_transform(cscaled_data)
print("T-SNE training finished")


#Plotting Phase
print("Beginning Plotting")

plt.figure(figsize=(10, 10))
plt.title("Fashion-MNIST")
plt.scatter(fscaled_data[:, 0], fscaled_data[:, 1], c=ftargetFrame, cmap='plasma')
plt.show()

plt.figure(figsize=(10, 10))
plt.title("one-hundred-plants-margin")
plt.scatter(sscaled_data[:, 0], sscaled_data[:, 1], c=stargetFrame, cmap='plasma')
plt.show()

plt.figure(figsize=(10, 10))
plt.title("isolet")
plt.scatter(cscaled_data[:, 0], cscaled_data[:, 1], c=ctargetFrame, cmap='plasma')
plt.show()


plt.figure(figsize=(10, 10))
plt.title("PCA - Fashion-MNIST")
plt.scatter(pcad_fdata[:, 0], pcad_fdata[:, 1], c=ftargetFrame, cmap='plasma')
plt.show()

plt.figure(figsize=(10, 10))
plt.title("PCA - one-hundred-plants-margin")
plt.scatter(pcad_sdata[:, 0], pcad_sdata[:, 1], c=stargetFrame, cmap='plasma')
plt.show()

plt.figure(figsize=(10, 10))
plt.title("PCA - isolet")
plt.scatter(pcad_cdata[:, 0], pcad_cdata[:, 1], c=ctargetFrame, cmap='plasma')
plt.show()


plt.figure(figsize=(10, 10))
plt.title("ICA - Fashion-MNIST")
plt.scatter(icad_fdata[:, 0], icad_fdata[:, 1], c=ftargetFrame, cmap='plasma')
plt.show()

plt.figure(figsize=(10, 10))
plt.title("ICA - one-hundred-plants-margin")
plt.scatter(icad_sdata[:, 0], icad_sdata[:, 1], c=stargetFrame, cmap='plasma')
plt.show()

plt.figure(figsize=(10, 10))
plt.title("ICA - isolet")
plt.scatter(icad_cdata[:, 0], icad_cdata[:, 1], c=ctargetFrame, cmap='plasma')
plt.show()


plt.figure(figsize=(10, 10))
plt.title("LDA - Fashion-MNIST")
plt.scatter(lda_fdata[:, 0], lda_fdata[:, 1], c=ftargetFrame, cmap='plasma')
plt.show()

plt.figure(figsize=(10, 10))
plt.title("LDA - one-hundred-plants-margin")
plt.scatter(lda_sdata[:, 0], lda_sdata[:, 1], c=stargetFrame, cmap='plasma')
plt.show()

plt.figure(figsize=(10, 10))
plt.title("LDA - isolet")
plt.scatter(lda_cdata[:, 0], lda_cdata[:, 1], c=ctargetFrame, cmap='plasma')
plt.show()


plt.figure(figsize=(10, 10))
plt.title("UMAP - Fashion-MNIST")
plt.scatter(ump_fdata[:, 0], ump_fdata[:, 1], c=ftargetFrame, cmap='plasma')
plt.show()

plt.figure(figsize=(10, 10))
plt.title("UMAP - one-hundred-plants-margin")
plt.scatter(ump_sdata[:, 0], ump_sdata[:, 1], c=stargetFrame, cmap='plasma')
plt.show()

plt.figure(figsize=(10, 10))
plt.title("UMAP - isolet")
plt.scatter(ump_cdata[:, 0], ump_cdata[:, 1], c=ctargetFrame, cmap='plasma')
plt.show()


plt.figure(figsize=(10, 10))
plt.title("T-SNE - Fashion-MNIST")
plt.scatter(tsn_fdata[:, 0], tsn_fdata[:, 1], c=ftargetFrame, cmap='plasma')
plt.show()

plt.figure(figsize=(10, 10))
plt.title("T-SNE - one-hundred-plants-margin")
plt.scatter(tsn_sdata[:, 0], tsn_sdata[:, 1], c=stargetFrame, cmap='plasma')
plt.show()

plt.figure(figsize=(10, 10))
plt.title("T-SNE - isolet")
plt.scatter(tsn_cdata[:, 0], tsn_cdata[:, 1], c=ctargetFrame, cmap='plasma')
plt.show()

print("Finished Plotting")

##I am resampling the data because the original dataset is too large
print("Beginning evaluation")
mres = resample(fscaled_data, n_samples= 200, random_state=345789)
ores = resample(sscaled_data, n_samples= 200, random_state=345789)
ires = resample(cscaled_data, n_samples= 200, random_state=345789)

m_pca_res = resample(pcad_fdata, n_samples= 200, random_state=345789)
o_pca_res = resample(pcad_sdata, n_samples= 200, random_state=345789)
i_pca_res = resample(pcad_cdata, n_samples= 200, random_state=345789)

m_ica_res = resample(icad_fdata, n_samples= 200, random_state=345789)
o_ica_res = resample(icad_sdata, n_samples= 200, random_state=345789)
i_ica_res = resample(icad_cdata, n_samples= 200, random_state=345789)

m_lda_res = resample(lda_fdata, n_samples= 200, random_state=345789)
o_lda_res = resample(lda_sdata, n_samples= 200, random_state=345789)
i_lda_res = resample(lda_cdata, n_samples= 200, random_state=345789)

m_umap_res = resample(ump_fdata, n_samples= 200, random_state=345789)
o_umap_res = resample(ump_sdata, n_samples= 200, random_state=345789)
i_umap_res = resample(ump_cdata, n_samples= 200, random_state=345789)

m_tsne_res = resample(tsn_fdata, n_samples= 200, random_state=345789)
o_tsne_res = resample(tsn_sdata, n_samples= 200, random_state=345789)
i_tsne_res = resample(tsn_cdata, n_samples= 200, random_state=345789)

pcaf_trust = trustworthiness(mres, m_pca_res)
pcas_trust = trustworthiness(ores, o_pca_res)
pcac_trust = trustworthiness(ires, i_pca_res)

icaf_trust = trustworthiness(mres, m_ica_res)
icas_trust = trustworthiness(ores, o_ica_res)
icac_trust = trustworthiness(ires, i_ica_res)

ldaf_trust = trustworthiness(mres, m_lda_res)
ldas_trust = trustworthiness(ores, o_lda_res)
ldac_trust = trustworthiness(ires, i_lda_res)

umapf_trust = trustworthiness(mres, m_umap_res)
umaps_trust = trustworthiness(ores, o_umap_res)
umapc_trust = trustworthiness(ires, i_umap_res)

tsnef_trust = trustworthiness(mres, m_tsne_res)
tsnes_trust = trustworthiness(ores, o_tsne_res)
tsnec_trust = trustworthiness(ires, i_tsne_res)


print("Trustworthiness with Fashion-MNIST: PCA=" + str(pcaf_trust) + ", FastICA=" + str(icaf_trust) + ", LDA=" + str(ldaf_trust) + ", T-SNE=" + str(tsnef_trust) + ", UMAP=" + str(umapf_trust))
print("Trustworthiness with one-hundred-plants-margin: PCA=" + str(pcas_trust) + ", FastICA=" + str(icas_trust) + ", LDA=" + str(ldas_trust) + ", T-SNE=" + str(tsnes_trust) + ", UMAP=" + str(umaps_trust))
print("Trustworthiness with Isolet: PCA=" + str(pcac_trust) + ", FastICA=" + str(icac_trust) + ", LDA=" + str(ldac_trust) + ", T-SNE=" + str(tsnec_trust) + ", UMAP=" + str(umapc_trust))
print("Finished evaluation")
