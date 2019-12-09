import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix

def normalize(df): 
	'''
	Takes: dataframe
	Returns: dataframe with all features normalized
	'''

	df_min = df.min()
	df_range = (df - df_min).max()
	df_scaled = (df - df_min) / df_range
	return df_scaled

def get_confusion_matrix(y_test, y_predict): 
	'''
	Takes: test true labels, predicted labels
	Returns: confusion matrix
	'''

	cm = np.array(confusion_matrix(y_test, y_predict, labels = [1, 0]))
	cm_df = pd.DataFrame(cm,
		index = ['is_cancer', 'is_healthy'],
		columns = ['predicted_cancer','predicted_healthy'])
	return cm_df 

def bootstrap_with_noise(df, m):
	'''
	Takes: df, int denoting how many multiples larger the new dataset will be
			(i.e. old_n * m = new_n)
	Returns: bootstrapped df with random noise
	'''

	bdf = df.sample(frac = m, replace = True, random_state = 123)
	noise = np.random.normal(0, 1, bdf.shape)
	noisy_bdf = bdf + noise
	noisy_bdf['target'] = bdf['target']

	return noisy_bdf