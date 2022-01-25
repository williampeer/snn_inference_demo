import matplotlib.pyplot as plt
import pyspike as spk
import numpy as np

# spike_trains = spk.load_spike_trains_from_txt("/home/william/repos/PySpike/test/PySpike_testdata.txt", edges=(0, 4000))
from pyspike import SpikeTrain

from data_util import get_spike_train_matrix
from ext_spike_metrics import get_label_free_isi_dist

model_trains = []
target_trains = []
for i in range(10):  # sim. 10 nodes
    model_trains.append(SpikeTrain(np.sort(4000. * np.random.random((50+20*i,))), 4000))
    target_trains.append(SpikeTrain(np.sort(4000. * np.random.random((50+20*i,))), 4000))

m0 = SpikeTrain(model_trains[0], 4000)
t0 = SpikeTrain(target_trains[0], 4000)
isi_profile_0_0 = spk.isi_profile(m0, t0)
isi_profile_n_n = spk.isi_profile(model_trains[-1], target_trains[-1])

# isi_profile = spk.isi_profile(model_trains)
x, y = isi_profile_0_0.get_plottable_data()
plt.plot(x, y, '--k')
print("ISI distance: %.8f" % isi_profile_0_0.avrg())
plt.show()

mt0 = get_spike_train_matrix(0, 4000, model_trains[0], np.zeros_like(model_trains[0]), np.array([0]))
tt0 = get_spike_train_matrix(0, 4000, target_trains[0], np.zeros_like(target_trains[0]), np.array([0]))
sut = get_label_free_isi_dist(mt0, tt0, 4000)
assert sut == isi_profile_0_0
