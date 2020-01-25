import sys
sys.path.append('../code')
import numpy as np
from KNN.KNN import KNN
from KMeans.KMeans import KMeans

X = np.array([[12, 39],[20, 36],[28, 30],[18, 52],[29, 54],[33, 46],[24, 55],[45, 59],[45, 63],[52, 70],[51, 66],[52, 63],[55, 58],[53, 23],[55, 14],[61,  8],[64, 19],[69,  7],[72, 24],[20, 50]])
y = np.array([0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0])


model = KNN()
model.fit(X, y, 3)
pred = model.predict(np.array([[12,40], [65, 5]]))
print(pred)
