from os.path import dirname, join

import numpy as np
import pandas as pd
import pandas.io.sql as psql
import sqlite3 as sql
from io import StringIO

from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox, row
from bokeh.models import ColumnDataSource, HoverTool, Div, LabelSet, Slider, Panel, Tabs, Range1d
from bokeh.models.widgets import MultiSelect, TextInput, Select, Button, Paragraph
from bokeh.io import curdoc
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral6
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

args = curdoc().session_context.request.arguments
datasetname = str(args.get('dsname')[0].decode('utf-8'))
#datasetname  = "churn.csv"
print("Dataset name is " +datasetname)

desc = Div(text="""
<h2 style="font-family="Arial">
Select the features to be included in the Gradient Boosting Model
</h2>

<p><a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html" target="_blank">Click here </a>for more information on the parameters </p>
""",width=1100)

df = pd.read_csv(datasetname)

y = df[df.columns[:1]].values.ravel()
df1 = df.drop(df.columns[:1],axis=1)

features = MultiSelect(title="Features",
               options=df.columns[1:].tolist())  
loss = Select(title="Loss:", value="deviance", options=["deviance", "exponential"])  
n_estimators = Slider(start=0, end = 30, value = 10, step =1, title="No of estimators:")
criterion = Select(title="Criterion:", value="friedman_mse", options=["’friedman_mse’", "mse","mae"])
max_depth = Slider(start=0, end=30, value=3, step=1,
                    title="Max_Depth")
warm_start = Select(title="Warm_start:", value="False", options=["True", "False"])   

'''
div = Div(text="""Your <a href="https://en.wikipedia.org/wiki/HTML">HTML</a>-supported text is initialized with the <b>text</b> argument.  The
remaining div arguments are <b>width</b> and <b>height</b>. For this example, those values
are <i>200</i> and <i>100</i> respectively.""",
width=200, height=100)
'''
target = Paragraph(text='',name = 'target')
target.text = "Target feature is " + str(df.columns[:1].tolist())
stats = Paragraph(text='',width=1000,name = 'Selected Features:')

#columns =['avg_dist', 'avg_rating_by_driver','avg_rating_of_driver','avg_surge','surge_pct','trips_in_first_30_days','luxury_car_user','weekday_pct','city_Astapor',"city_KingsLanding",'city_Winterfell','phone_Android','phone_no_phone']
#columns = ['luxury_car_user','avg_dist','city_Astapor',"city_KingsLanding",'phone_Android','phone_iPhone']

#df1 = pd.DataFrame(df, columns=columns)
#y = df['churn']
y = df[df.columns[:1]].values.ravel()
df1 = df.drop(df.columns[:1],axis=1)
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

x_train_original,x_test_original,y_train_original,y_test_original=train_test_split(X_new,y,test_size=0.25)
#For standardizing data

#clf = svm.LinearSVC(random_state=0)
clf = GradientBoostingClassifier()
clf.fit(x_train_original,y_train_original)
predictions=clf.predict(x_test_original)
#print("Accuracy =", accuracy_score(y_test_original,predictions))
#print(np.unique(predictions))
tn, fp, fn, tp = confusion_matrix(y_test_original,predictions).ravel()


fruits = ['True Positive', 'False Positive', 'True Negative', 'False Negative']
#fruits = [tp, fp, tn, fn]
#counts = [0, 0, 0, 0]
counts = [tp, fp, tn, fn]

source = ColumnDataSource(data=dict(fruits=fruits, counts=counts))

p = figure(x_range=fruits, plot_height=350, title="Counts")
p.vbar(x='fruits', top='counts', width=0.9, source=source, legend="fruits",
       line_color='white',fill_color=factor_cmap('fruits', palette=Spectral6, factors=fruits))
p.title.text = "Model Accuracy %f" % accuracy_score(y_test_original,predictions)
labels = LabelSet(x='fruits', y='counts', text='counts', level='glyph',
        x_offset=-15, y_offset=0, source=source, render_mode='canvas')
p.yaxis.axis_label = "Counts"
p.add_layout(labels)
      
tab1 = Panel(child=p, title="Accuracy Scores")

p1 = figure(plot_height=350, title="PR Curve")
p1.x_range = Range1d(0,1)
p1.y_range = Range1d(0,1)
p1.yaxis.axis_label = "Precision"
p1.xaxis.axis_label = "Recall"
#p1.line([0],[0],name ="line2")

y_score = clf.predict_proba(x_test_original)[:,1]
precision, recall, _ = precision_recall_curve(y_test_original, y_score)
p1.line(precision, recall, line_width=2,line_alpha=0.6,name ="line2")
average_precision = average_precision_score(y_test_original, predictions)
p1.title.text = "Average Precision Score %f" % average_precision

tab2 = Panel(child=p1, title="PR Curve")

tabs = Tabs(tabs=[ tab1, tab2 ])
#p.title.text = "Model Accuracy %f" % accuracy_score(y_test_original,predictions)


def update():
    line = p1.select_one({'name': 'line2'})
    p1.renderers.remove(line)
    line.visible = False
    precision = 0
    recall = 0
    p1.line(precision,recall,line_alpha=0)
    fval = features.value
    print(fval)  
    stats.text = "Selected features : " + str(fval)  
    df1 = pd.DataFrame(df, columns=fval)
    #y = df['churn']
    x_train_original,x_test_original,y_train_original,y_test_original=train_test_split(df1,y,test_size=0.25)
    #clf = svm.SVC(kernel=kern)

    loss1 = loss.value
    criterion1 = criterion.value
    n_estimators1 = int(n_estimators.value)
    max_depth1 = int(max_depth.value) 

    if (warm_start.value == 'True'):
        warm_start1 = True
    else:
        warm_start1 = False

        
    
    clf = GradientBoostingClassifier(warm_start = warm_start1, loss = loss1, criterion = criterion1, n_estimators = n_estimators1, max_depth = max_depth1)
    clf.fit(x_train_original,y_train_original)
    predictions=clf.predict(x_test_original)
    print("Accuracy =", accuracy_score(y_test_original,predictions))
    y_score = clf.predict_proba(x_test_original)[:,1]
    precision, recall, _ = precision_recall_curve(y_test_original, y_score)
    p1.line(precision, recall, line_width=2,line_alpha=0.6,name ="line2")
    average_precision = average_precision_score(y_test_original, predictions)
    p1.title.text = "Average Precision Score %f" % average_precision
    #print(np.unique(predictions))
    tn, fp, fn, tp = confusion_matrix(y_test_original,predictions).ravel()
    source.data =dict(fruits=fruits, counts=[tp, fp, tn, fn])
    p.title.text = "Model Accuracy %f" % accuracy_score(y_test_original,predictions)

controls = [features, warm_start, loss, criterion, n_estimators, max_depth]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

sizing_mode = 'fixed'  # 'scale_width' also looks nice with this example

inputs = widgetbox(*controls, sizing_mode=sizing_mode)
l = layout([
    [desc],
    [row(target,stats)],
    [inputs,tabs]
],sizing_mode= sizing_mode)


#update()  # initial load of the data
 
curdoc().add_root(l)
curdoc().title = "Churn" 