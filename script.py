import numpy as np

d = np.load("./patches/patches_32.npz", allow_pickle=True)
np.save("./patches/X_32.npy", d['X'])
np.save("./patches/y_32.npy", d['y'].astype(np.int64))
np.save("./patches/subj_32.npy", d['subj'].astype(str))
print("Готово:")
print("  X_32.npy   — заливать на Drive (это главный, ~6 ГБ)")
print("  y_32.npy   — маленький")
print("  subj_32.npy — маленький")