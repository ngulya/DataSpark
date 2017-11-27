import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import os
import datetime
import copy
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
import scipy.sparse.csr as csr
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from keras.models import model_from_json
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
warnings.simplefilter('ignore')

np.random.seed(42)
ctv = CountVectorizer()
TIMEZONE = 7200#MY_TIMEZONE - GMT = 7200 sec 
issues = pd.read_csv("./issues.csv")
commits = pd.read_csv("./commits.csv")
tmp = pd.DataFrame({'key':[]})
tmp['key'][0] = 'start'
_z = 0

def output_res(Y_test, xx, result, s):
	plt.scatter(xx, Y_test, color='r', linewidth=3, label='Test')
	plt.scatter(xx, result, color='b', linewidth=0.1, label='Prediction')
	plt.ylabel('Target label')
	plt.xlabel('Line number in dataset')
	plt.legend(loc=5)
	plt.text(0,8,"%s"%s)
	plt.text(0,7.5,"MSE = %s"%mean_squared_error(Y_test, result))
	plt.text(0,7,"accuracy = %f"% accuracy_score(Y_test, result))
	plt.show()
	df_cm = pd.DataFrame(metrics.confusion_matrix(Y_test, result), index = [i for i in "54321"],columns = [i for i in "12345"])
	plt.figure(figsize = (10,7))
	sn.heatmap(df_cm, annot=True)
	plt.show()

def normalize_array(ar):
	maxar = max(ar)
	minar = min(ar)
	r = (ar - minar)/(maxar - minar)
	return r

def normalize_Y_categoty(v, YN, x):
	x -= 1
	if v == 0:
		YN[x + 1] = 1
		YN[x + 2] = 0
		YN[x + 3] = 0
		YN[x + 4] = 0
		YN[x + 5] = 0
	if v == 1:
		YN[x + 1] = 0
		YN[x + 2] = 1
		YN[x + 3] = 0
		YN[x + 4] = 0
		YN[x + 5] = 0
	if v == 3:
		YN[x + 1] = 0
		YN[x + 2] = 0
		YN[x + 3] = 1
		YN[x + 4] = 0
		YN[x + 5] = 0
	if v == 5:
		YN[x + 1] = 0
		YN[x + 2] = 0
		YN[x + 3] = 0
		YN[x + 4] = 1
		YN[x + 5] = 0
	if v == 10:
		YN[x + 1] = 0
		YN[x + 2] = 0
		YN[x + 3] = 0
		YN[x + 4] = 0
		YN[x + 5] = 1
	return YN

def strtonum(v):
	v = v.lower()
	res = 0
	for x in v:
		res += ord(x)
	return res

def short_spark(v):
	v = v[6:]
	return int(v)

def short_date(v):
	if v != 0:
		v = v[0:4] + v[5:7] + v[8:10]
		if v[0] != '2' or v[1] != '0':
			ret = '20' + v[2:]
			return int(ret)
	return int(v)

def do_date(v):
	date_time = datetime.datetime.fromtimestamp(v - TIMEZONE).strftime('%Y-%m-%d')
	return date_time


def do_hours(v):
	date_time = datetime.datetime.fromtimestamp(v - TIMEZONE).strftime('%H')
	return int(date_time)

def people_commit_per_day(r):
	ppl = 0
	for v in r:
		if v != 0:
			ppl += 1
	return (ppl)

def num_commits(r):
	ppl = 0
	com = 0
	for v in r:
		if v != 0:
			ppl += 1
			com += v
	return (int(com/ppl))

def average_severity(v):

	array_sev = v.index.get_values()
	sums = 0
	i = 0
	sz = 0
	for x in v:
		sums += x * array_sev[i]
		sz += x
		i += 1
	return int(sums/sz)

def return_key(v):
	global _z
	global tmp

	print(_z)
	_z += 1
	if v['key'] == ' ':
		valu = (issues['key'][issues['updated'] == v['date']]).get_values()
		valc = (issues['key'][issues['created'] == v['date']]).get_values()
		valr = (issues['key'][issues['resolved'] == v['date']]).get_values()
		for sprk in valu:
			if  (tmp['key'] != sprk).all() and (commits['key'] != sprk).all():
				tmp['key'][_z] = sprk
				return sprk
		for sprk in valc:
			if  (tmp['key'] != sprk).all() and (commits['key'] != sprk).all():
				tmp['key'][_z] = sprk
				return sprk
		for sprk in valr:
			if  (tmp['key'] != sprk).all() and (commits['key'] != sprk).all():
				tmp['key'][_z] = sprk
				return sprk
	return v['key']


def return_type(v):
	global _z
	print(_z)
	_z += 1
	if v['key'] != ' ':
		val = (issues['issuetype'][issues['key'] == v['key']]).get_values()
		if len(val) == 1:
			return val[0]
	return v['issuestype']

def return_priority(v):
	global _z
	print(_z)
	_z -= 1
	if v['key'] != ' ':
		val = (issues['priority'][issues['key'] == v['key']]).get_values()
		if len(val) == 1:
			return val[0]
	return v['priority']

def return_severity(v):
	global _z

	print(_z)
	_z -= 1
	if v != ' ':
		l = issues['key'] == v
		sever = issues['severity'][l].get_values()
		if len(sever) == 1:
			return int(sever)
	return -1

def return_simple_message(v):

	str_ret ='St '#for trash
	n = v.split('\n')
	for x in n:
		if not x.startswith('[') and not x.startswith('*') and not x.startswith('!') and not x.startswith('##'):
			str_ret += x + ' '
		if x.startswith('## How') or x.startswith('### How'):
			str_ret += 'BInGO '
		if x.startswith('Author') or x.startswith('Closes'):
			break
	return str_ret

def sparktoint(v):
	return int(v[6:])

def return_random_num():
	forar = [0,1,3,5,10]
	return forar[np.random.randint(0, 5)]

def Graph_(data):
	Grph = data.groupby(['severity', 'committer_name']).size().unstack().fillna(0)
	Grph = Grph.apply(average_severity)
	Grph.plot(title='Severity for each committer')
	plt.show()


	Grph = data.groupby(['year', 'severity']).size().unstack().fillna(0)
	Grph.plot(title='severity per year')
	plt.show()

	Grph = data.groupby(['week', 'severity']).size().unstack().fillna(0)
	Grph.plot(title='severity per week')
	plt.show()

	Grph = data.groupby(['weekday', 'severity']).size().unstack().fillna(0)
	Grph.plot(title='severity per weekday')
	plt.show()

	Grph = data.groupby(['hours', 'severity']).size().unstack().fillna(0)
	Grph.plot(title='severity per hours')
	plt.show()

	Grph = data.groupby(['issuestype', 'severity']).size().unstack().fillna(0)
	Grph.plot(title='severity per issuestype')
	plt.show()

	Grph = data.groupby(['priority', 'severity']).size().unstack().fillna(0)
	Grph.plot(title='severity per priority')
	plt.show()


	Grph = data.groupby(['time_offset', 'severity']).size().unstack().fillna(0)
	Grph.plot(title='severity per time_offset')
	plt.show()



###MAIN
if os.path.exists("./data_frame.csv"):
	data = pd.read_csv("./data_frame.csv")
else:
	print("Create data_frame.csv  0-39k-0")
	commits = commits.drop(['cid', 'tree_id'], axis = 1)
	commits['severity'] = pd.DataFrame({'severity':[]})
	commits['date'] = pd.DataFrame({'date':[]})
	commits['issuestype'] = pd.DataFrame({'issuestype':[]})
	commits['priority'] = pd.DataFrame({'priority':[]})
	commits['year'] = pd.DataFrame({'year':[]})
	commits['week'] = pd.DataFrame({'week':[]})
	commits['weekday'] = pd.DataFrame({'weekday':[]})
	commits['hours'] = pd.DataFrame({'hours':[]})

	issues['updated'] = issues['updated'].fillna(0)
	issues['updated'] = issues['updated'].apply(short_date)
	issues['created'] = issues['created'].fillna(0)
	issues['created'] = issues['created'].apply(short_date)
	issues['resolved'] = issues['resolved'].fillna(0)
	issues['resolved'] = issues['resolved'].apply(short_date)

	commits['date'] = commits['time'].apply(do_date)
	commits['hours'] = commits['time'].apply(do_hours)
	commits['week'] = commits['date'].apply(pd.to_datetime).dt.week
	commits['weekday'] =  commits['date'].apply(pd.to_datetime).dt.weekday
	commits['year'] =  commits['date'].apply(pd.to_datetime).dt.year
	commits['date'] = commits['date'].apply(short_date)

	commits['key'] = commits['key'].fillna(' ')
	commits['key'] = commits.apply(return_key, axis = 1)

	commits['issuestype'] = commits['issuestype'].fillna(' ')
	commits['issuestype'] = commits.apply(return_type, axis = 1)

	commits['priority'] = commits['priority'].fillna(' ')
	commits['priority'] = commits.apply(return_priority, axis = 1)

	commits['severity'] = commits['key'].apply(return_severity)

	commits['severity'] = commits['severity'].fillna(-1)
	commits = commits[commits['severity'] != -1]

	commits['priority'] = commits['priority'].apply(strtonum)
	commits['issuestype'] = commits['issuestype'].apply(strtonum)
	commits['author_name'] = commits['author_name'].apply(strtonum)
	commits['committer_name'] = commits['committer_name'].apply(strtonum)
	commits['committer_email'] = commits['committer_email'].apply(strtonum)
	commits['author_email'] = commits['author_email'].apply(strtonum)
	commits['key'] = commits['key'].apply(sparktoint)
	commits['message_encoding'] = commits['message_encoding'].apply(return_simple_message)
	
	for_mess = ctv.fit_transform(commits['message_encoding'])
	commits['message_encoding'] = list(for_mess)
	
	save = commits
	save.index = commits['key']
	save =  save.drop(['key'], axis = 1)
	save.to_csv('./data_frame.csv')
	print("ok")

	data = commits

Graph_(data)

###DATA for test
target = pd.DataFrame({'severity':[]})
target['severity'] = data['severity']
savedat = data
data = data.drop(['key', 'author_email', 'committer_email', 'time', 'message_encoding', 'severity'], axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(data, target,test_size =  0.2)
Y_test = Y_test.as_matrix().flatten()
Y_train = Y_train.as_matrix().flatten()
X_test = X_test.as_matrix()
X_train = X_train.as_matrix()
xx = [i for i in range(Y_test.shape[0])]

##Classification
model = RandomForestClassifier(n_estimators = 4, n_jobs=-1, max_features=5)
model.fit(X_train, Y_train)
result = model.predict(X_test)
output_res(Y_test, xx, result, "RandomForest")


model = DecisionTreeClassifier(max_features=5)
model.fit(X_train, Y_train)
result = model.predict(X_test)
output_res(Y_test, xx, result, "DecisionTree")


model = KNeighborsClassifier(n_neighbors = 1,leaf_size = 20, n_jobs = -1, p =1)#The best 1 neigh parameters
model.fit(X_train, Y_train)
result = model.predict(X_test)
output_res(Y_test, xx, result, "KNeighbors")


##NEURO
XN_test = copy.deepcopy(X_test)
XN_test = XN_test.astype('float32')
x = 0
while x < 10:
	XN_test[:,x] = normalize_array(XN_test[:,x])
	x += 1
XN_train = copy.deepcopy(X_train)
XN_train = XN_train.astype('float32')

x = 0
while x < 10:
	XN_train[:,x] = normalize_array(XN_train[:,x])
	x += 1


YN_train = np.arange(len(Y_train) * 5)
x = 0
for row in Y_train: 
	YN_train = normalize_Y_categoty(row, YN_train,x)
	x += 5
YN_train.shape = (len(Y_train), 5)
YN_test = copy.deepcopy(Y_test)
np.random.seed(42)
yn = input("Download Neuro Y-Yes(Recommend) N-No: ")
if yn == 'y' or yn == 'Y':
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	modelN = model_from_json(loaded_model_json)
	modelN.load_weights("model.h5")
else:
	modelN = Sequential()
	modelN.add(Dense(10, input_dim = 10, activation="relu", kernel_initializer="normal"))
	modelN.add(Dense(5, activation="softmax", kernel_initializer="normal"))

	modelN.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
	print(modelN.summary())
	modelN.fit(XN_train, YN_train,epochs=300, verbose=1)
	resultN = modelN.predict(XN_test, batch_size=32)
	yn = input("Save Neuro Y-Yes N-No: ")
	if (yn == 'y' or yn == 'Y'):
		model_json = modelN.to_json()
		with open("model.json", "w") as json_file:
			json_file.write(model_json)
		modelN.save_weights("model.h5")
	

resultN = modelN.predict(XN_test, batch_size=32)
resultN.shape = (len(Y_test) * 5, 1)
normres = np.arange(len(Y_test))
x = 0
n_r = 0
lenres = len(resultN)
while x < lenres:
	y = 0
	while y < 5 and x < lenres:
		if resultN[x] >= 0.4 and y == 0:
			normres[n_r] = 0
			n_r += 1
			x += 5 - y
			break
		elif resultN[x] >= 0.4 and y == 1:
			normres[n_r] = 1
			n_r += 1
			x += 5 - y
			break
		elif resultN[x] >= 0.4 and y == 2:
			normres[n_r] = 3
			n_r += 1
			x += 5 - y
			break
		elif resultN[x] >= 0.4 and y == 3:
			normres[n_r] = 5
			n_r += 1
			x += 5 - y
			break
		elif resultN[x] >= 0.4 and y == 4:
			normres[n_r] = 10
			n_r += 1
			x += 5 - y
			break
		else:
			x+=1
			y+=1
		if y == 5:
			normres[n_r] = return_random_num()
			n_r += 1

x = 0
output_res(YN_test, xx, normres, "Neuro")
