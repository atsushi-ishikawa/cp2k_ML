import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import argparse

matplotlib.rcParams['backend'] = 'TkAgg'

np.random.seed(10)

def drop_duplicated_sample(df, key=None):
	duplicated = df[key].apply(tuple).duplicated()
	df = df[~duplicated]
	return df

def regression(df, do_plot=True):
	from sklearn.linear_model import LinearRegression, Lasso
	from sklearn.preprocessing import StandardScaler
	from sklearn.pipeline import Pipeline
	from sklearn.model_selection import train_test_split, GridSearchCV
	from sklearn.metrics import mean_squared_error

	# drop duplicate, identified with surface symbols (list)
	df = drop_duplicated_sample(df, key="surf_symbols")

	x = df.drop(["surf_formula", "surf_symbols", "ads_energy"], axis=1)
	y = df["ads_energy"]

	cv = 5
	test_size = 1.0 / cv

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

	scaler = StandardScaler()
	#method = LinearRegression()
	method = Lasso()
	pipe = Pipeline([("scl", scaler), ("reg", method)])
	param_grid = {"reg" + "__alpha": list(10**np.arange(-2, 2, 1.0))}
	grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv)
	grid.fit(x_train, y_train)

	print(pd.DataFrame({"name": x.columns, "Coef": grid.best_estimator_.named_steps["reg"].coef_}))
	print("Training set score: {:.3f}".format(grid.score(x_train, y_train)))
	print("Test set score: {:.3f}".format(grid.score(x_test, y_test)))
	print("RMSE: {:.3f}".format(np.sqrt(mean_squared_error(y_test, grid.predict(x_test)))))

	if do_plot:
		fig, ax = plt.subplots(figsize=(6, 6))
		seaborn.regplot(x=grid.predict(x), y=y.values,
			scatter_kws={"color": "navy", 'alpha': 0.3}, line_kws={"color": "navy"})
		ax.set_xlabel("Predicted value")
		ax.set_ylabel("True value")
		fig.tight_layout()
		#plt.show()
		fig.savefig("plot.png")
		plt.close()

# ---- start
parser = argparse.ArgumentParser()
parser.add_argument("--jsonfile", help="json file with data", default="data.json")
args = parser.parse_args()
jsonfile = args.jsonfile

df = pd.read_json(jsonfile, orient="records", lines=True)

# pairplot
plot = seaborn.pairplot(df)
#plt.show()
plt.savefig("pairplot.png")
plt.close()

regression(df)
