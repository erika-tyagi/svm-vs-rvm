import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix

def normalize(df): 
	df_min = df.min()
	df_range = (df - df_min).max()
	df_scaled = (df - df_min) / df_range
	return df_scaled

def get_confusion_matrix(y_test, y_predict): 
	cm = np.array(confusion_matrix(y_test, y_predict, labels = [1, 0]))
	cm_df = pd.DataFrame(cm,
		index = ['is_cancer', 'is_healthy'],
		columns = ['predicted_cancer','predicted_healthy'])
	return cm_df 
