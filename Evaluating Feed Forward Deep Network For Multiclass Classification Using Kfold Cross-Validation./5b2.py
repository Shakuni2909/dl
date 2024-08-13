#prac5b2nd
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
dataset=pandas.read_csv("flowers.csv",header=None)
dataset1=dataset.values
X=dataset1[1:,0:4].astype(float)
Y=dataset1[1:,4]
print(Y)
encoder=LabelEncoder()
encoder.fit(Y)
encoder_Y=encoder.transform(Y)
print(encoder_Y)
dummy_Y=np_utils.to_categorical(encoder_Y)
print(dummy_Y)
def baseline_model():
  model=Sequential()
  model.add(Dense(8,input_dim=4,activation='relu'))
  model.add(Dense(3,activation='softmax'))
  model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  return model
estimator=KerasClassifier(build_fn=baseline_model,epochs=5,batch_size=5)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
