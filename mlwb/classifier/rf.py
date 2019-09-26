import os
import numpy as np
import pandas as pd
import pandas.io.sql as psql
import sqlite3 as sql
from io import StringIO

from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox, row
from bokeh.models import ColumnDataSource, HoverTool, Div, LabelSet, Slider, Tabs, Panel, Range1d
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import MultiSelect, TextInput, Select, Button, Paragraph
from bokeh.io import curdoc
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, auc
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score, roc_curve
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral6
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from bokeh import events

args = curdoc().session_context.request.arguments
datasetname = str(args.get('dsname')[0].decode('utf-8'))
#datasetname  = "cancer.csv"
print("Dataset name is " +datasetname)
desc = Div(text="""
<h2 style="font-family="Arial">
Select the features to be included in the Random Forest Model
</h2>

<p><a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier" target="_blank">Click here </a>for more information on the parameters </p>
""",width=1100)

df = pd.read_csv(os.path.join('..', 'data', datasetname))

y = df[df.columns[:1]].values.ravel()
df1 = df.drop(df.columns[:1],axis=1)

features = MultiSelect(title="Features",
               options=df.columns[1:].tolist())
criterion = Select(title="Criterion:", value="gini", options=["gini", "entropy"])
max_depth = Slider(start=0, end=50, value=10, step=1,
                    title="Max_Depth")
n_estimators = Slider(start=0, end = 30, value = 10, step =1, title="No of estimators:")
bootstrap = Select(title="Bootstrap:", value="True", options=["True", "False"])   
oob_score = Select(title="oob_score:", value="False", options=["True", "False"])   
warm_start = Select(title="Warm_start:", value="False", options=["True", "False"])   


target = Paragraph(text='',name = 'target')
target.text = "Target feature is " + str(df.columns[:1].tolist())
stats = Paragraph(text='',width=1000,name = 'Selected Features:')

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
clf = RandomForestClassifier()
clf.fit(x_train_original,y_train_original)
predictions=clf.predict(x_test_original)
#print("Accuracy =", accuracy_score(y_test_original,predictions))
#print(np.unique(predictions))
tn, fp, fn, tp = confusion_matrix(y_test_original,predictions,labels=[0,1]).ravel()


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
auc_score = auc(recall,precision)
p1.title.text = "AUC PR %f" % auc_score

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
    crit = criterion.value
    maxd = max_depth.value
    df1 = pd.DataFrame(df, columns=fval)
    #y = df['churn']
    x_train_original,x_test_original,y_train_original,y_test_original=train_test_split(df1,y,test_size=0.25)
    #clf = svm.SVC(kernel=kern)
    n_estimators1 = int(n_estimators.value)
    if (bootstrap.value == 'True'):
        bootstrap1 = True
    else:
        bootstrap1 = False

    if (oob_score.value == 'True'):
        oob_score1 = True
        bootstrap1 = True
    else:
        oob_score1 = False

    if (warm_start.value == 'True'):
        warm_start1 = True
    else:
        warm_start1 = False
    
    clf = RandomForestClassifier(criterion=crit, max_depth = maxd, n_estimators = n_estimators1, bootstrap = bootstrap1, warm_start = warm_start1, oob_score = oob_score1)
    clf.fit(x_train_original,y_train_original)
    predictions=clf.predict(x_test_original)
    print("Accuracy =", accuracy_score(y_test_original,predictions))
    y_score = clf.predict_proba(x_test_original)[:,1]
    precision, recall, _ = precision_recall_curve(y_test_original, y_score)
    p1.line(precision, recall, line_width=2,line_alpha=0.6,name ="line2")
    average_precision = average_precision_score(y_test_original, predictions)
    auc_score = auc(recall,precision)
    p1.title.text = "AUC PR %f" % auc_score
    #print(np.unique(predictions))
    tn, fp, fn, tp = confusion_matrix(y_test_original,predictions,labels=[0,1]).ravel()
    source.data =dict(fruits=fruits, counts=[tp, fp, tn, fn])
    p.title.text = "Model Accuracy %f" % accuracy_score(y_test_original,predictions)
    

controls = [features, criterion, max_depth, n_estimators, bootstrap, oob_score, warm_start]
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