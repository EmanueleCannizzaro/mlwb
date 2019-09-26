import os
import numpy as np
import pandas as pd
import pandas.io.sql as psql
import sqlite3 as sql
from io import StringIO

from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox, row
from bokeh.models import ColumnDataSource, HoverTool, Div, LabelSet, Slider, Tabs, Panel, Range1d
from bokeh.models.widgets import MultiSelect, TextInput, Select, Button, Paragraph
from bokeh.io import curdoc
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score, mean_squared_error, r2_score
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral6
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression

args = curdoc().session_context.request.arguments
datasetname = str(args.get('dsname')[0].decode('utf-8'))
#datasetname  = "boston.csv"
print("Dataset name is " + datasetname)

desc = Div(text="""
<h2 style="font-family="Arial">
Select the features to be included in the Linear Regression Model
</h2>

<p><a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html" target="_blank">Click here </a>for more information on the parameters </p>
""",width=1100)


df = pd.read_csv(os.path.join('..', 'data', datasetname))

y = df[df.columns[:1]].values.ravel()
df1 = df.drop(df.columns[:1],axis=1)

target = Paragraph(text='',name = 'target')
target.text = "Target feature is " + str(df.columns[:1].tolist())

features = MultiSelect(title="Features",
               options=df.columns[1:].tolist())
kfold = Slider(start=2, end=10, value=5, step=1,title="No of folds")
fit_intercept = Select(title="Fit_intercept:", value="True", options=["True", "False"])   
normalize = Select(title="Normalize:", value="False", options=["True", "False"])   
copy_X = Select(title="Copy_X:", value="True", options=["True", "False"])   

stats = Paragraph(text='',width = 1000,height=50,name = 'Selected Features:')
stats2 =  Div(text='<h3>Results:</h3>',)

#columns =['avg_dist', 'avg_rating_by_driver','avg_rating_of_driver','avg_surge','surge_pct','trips_in_first_30_days','luxury_car_user','weekday_pct','city_Astapor',"city_KingsLanding",'city_Winterfell','phone_Android','phone_no_phone']
#columns = ['luxury_car_user','avg_dist','city_Astapor',"city_KingsLanding",'phone_Android','phone_iPhone']

#df1 = pd.DataFrame(df, columns=columns)
#y = df['churn']
y = df[df.columns[:1]].values.ravel()
df1 = df.drop(df.columns[:1],axis=1)
''' 
selector = SelectKBest(chi2, k=5).fit(df1, y)
X_new = selector.transform(df1)
mask = selector.get_support() #list of booleans
new_features = [] # The list of your K best features

for bool, feature in zip(mask, df.columns[1:].tolist()):
    if bool:
        new_features.append(feature)

#print(new_features)
features.value = new_features
stats.text = "Top 5 features according to Select K Best (Chi2) : " + str(new_features)
 '''
#print(new_features)


x_train_original,x_test_original,y_train_original,y_test_original=train_test_split(df1,y,test_size=0.25)

clf = LinearRegression()
clf.fit(x_train_original,y_train_original)
predictions=clf.predict(x_test_original)
scores = cross_val_score(clf,df1,y,cv=5,scoring='neg_mean_squared_error')

stats2.text += "Mean Squared Error: %.2f" % mean_squared_error(y_test_original, predictions) + '</br>'
stats2.text += " Variance score: %.2f" % r2_score(y_test_original, predictions) + '</br>'
stats2.text += " Cross Validation score: %.2f " % scores.mean()


''' p1 = figure(plot_height=350,title="PR Curve")
p1.x_range = Range1d(0,1)
p1.y_range = Range1d(0,1)
p1.line([0],[0],name ="line2")

tab1 = Panel(child=p1, title="PR Curve")
tabs = Tabs(tabs=[ tab1 ])
        '''

def update():
    ''' line = p1.select_one({'name': 'line2'})
    p1.renderers.remove(line)
    line.visible = False
    precision = 0
    recall = 0
    p1.line(precision,recall,line_alpha=0) '''
    fval = features.value
    print(fval)  
    stats.text = "Selected features : " + str(fval)  
    fit_intercept1 = fit_intercept.value
    normalize1 = normalize.value
    copy_X1 = copy_X.value
    df1 = pd.DataFrame(df, columns=fval)
    #y = df['churn']
    x_train_original,x_test_original,y_train_original,y_test_original=train_test_split(df1,y,test_size=0.25)
    #clf = svm.SVC(kernel=kern)
   
    clf = LinearRegression(fit_intercept = fit_intercept1, normalize = normalize1, copy_X = copy_X1)
    clf.fit(x_train_original,y_train_original)
    predictions=clf.predict(x_test_original)
    print("Mean squared error: %.2f"
      % mean_squared_error(y_test_original, predictions))
    print("Variance score: %.2f"
      % r2_score(y_test_original, predictions))
    #print("Accuracy =", accuracy_score(y_test_original,predictions))
    #average_precision = average_precision_score(y_test_original, predictions)
    #print("Average Precision: %.2f" % average_precision)
    #precision, recall, _ = precision_recall_curve(y_test_original, predictions)
    scores = cross_val_score(clf,df1,y,cv=int(kfold.value),scoring='neg_mean_squared_error')
    ''' p1.line(precision, recall, line_width=2,line_alpha=0.6,name ="line2")
    p1.title.text = "Mean Squared Error: %.2f" % mean_squared_error(y_test_original, predictions) 
    p1.title.text += " Variance score: %.2f" % r2_score(y_test_original, predictions)
    p1.title.text += " Average Precision: %.2f" % average_precision '''
    stats2.text = "<h3>Results:</h3>"
    stats2.text += "Mean Squared Error: %.2f" % mean_squared_error(y_test_original, predictions) + '</br>'
    stats2.text += " Variance score: %.2f" % r2_score(y_test_original, predictions) + '</br>'
    stats2.text += " Cross Validation score: %.2f " % scores.mean()


controls = [features,kfold, fit_intercept, normalize, copy_X]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

sizing_mode = 'fixed'  # 'scale_width' also looks nice with this example

inputs = widgetbox(*controls, sizing_mode=sizing_mode)
l = layout([
    [desc],
    [row(target,stats)],
    [inputs,stats2]
],sizing_mode= sizing_mode)

#update()  # initial load of the data
 
curdoc().add_root(l)
curdoc().title = "Churn" 
