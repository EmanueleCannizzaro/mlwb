import os
import numpy as np
import pandas as pd
import pandas.io.sql as psql
import sqlite3 as sql
from io import StringIO

from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox
from bokeh.models import ColumnDataSource, HoverTool, Div, LabelSet, Slider, Tabs, Panel, Range1d
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
from sklearn.gaussian_process import GaussianProcessClassifier

args = curdoc().session_context.request.arguments
datasetname = str(args.get('dsname')[0].decode('utf-8'))
#datasetname  = "churn.csv"
print("Dataset name is " +datasetname)
desc = Div(text="""<h1>An Interactive Explorer for Machine Learning datasets</h1>

<p>
Select the features to be included in the Gradient Boosting Model
</p>
<p>
Prepared by <b>Adil Khan</b>.
</p>
<p><a href="http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier" target="_blank">More information on the parameters</a></p>
<br/>
""")



#obj = client.get_object(Bucket='my-bucket', Key='churn.csv')
#df = pd.read_csv(obj['Body'])

#df = pd.read_csv('/Users/adilkhan/Documents/CS Fall 16/CS297/Bokeh-Demo/EmbedWebsite/cancer.csv')
#df = pd.read_csv('http://s3.amazonaws.com/cs297-mlplayground/'+datasetname)
df = pd.read_csv(os.path.join('..', 'data', datasetname))

y = df[df.columns[:1]].values.ravel()
df1 = df.drop(df.columns[:1],axis=1)

features = MultiSelect(title="Features",
               options=df.columns[1:].tolist())  
warm_start = Select(title="Warm_start:", value="False", options=["True", "False"])   
copy_X_train = Select(title="Copy_X_train:", value="True", options=["True", "False"]) 

'''
div = Div(text="""Your <a href="https://en.wikipedia.org/wiki/HTML">HTML</a>-supported text is initialized with the <b>text</b> argument.  The
remaining div arguments are <b>width</b> and <b>height</b>. For this example, those values
are <i>200</i> and <i>100</i> respectively.""",
width=200, height=100)
'''
stats = Paragraph(text='', width=800, height = 200,name = 'Selected Features:')

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

stats.text = str(new_features)

x_train_original,x_test_original,y_train_original,y_test_original=train_test_split(X_new,y,test_size=0.25)
#For standardizing data

#clf = svm.LinearSVC(random_state=0)
clf = GaussianProcessClassifier()
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

p = figure(x_range=fruits, plot_height=350, toolbar_location=None, title="Counts")
p.vbar(x='fruits', top='counts', width=0.9, source=source, legend="fruits",
       line_color='white',fill_color=factor_cmap('fruits', palette=Spectral6, factors=fruits))

labels = LabelSet(x='fruits', y='counts', text='counts', level='glyph',
        x_offset=-15, y_offset=0, source=source, render_mode='canvas')
p.add_layout(labels)      
tab1 = Panel(child=p, title="Accuracy Scores")

p1 = figure(plot_height=350, title="PR Curve")
p1.x_range = Range1d(0,1)
p1.y_range = Range1d(0,1)
p1.line([0],[0],name ="line2")
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
    stats.text = str(fval)  
    df1 = pd.DataFrame(df, columns=fval)
    #y = df['churn']
    x_train_original,x_test_original,y_train_original,y_test_original=train_test_split(df1,y,test_size=0.25)
    #clf = svm.SVC(kernel=kern)

    if (warm_start.value == 'True'):
        warm_start1 = True
    else:
        warm_start1 = False

    if (copy_X_train.value == 'True'):
        copy_X_train1 = True
    else:
        copy_X_train1 = False    
    
    clf = GaussianProcessClassifier(warm_start = warm_start1, copy_X_train = copy_X_train1)
    clf.fit(x_train_original,y_train_original)
    predictions=clf.predict(x_test_original)
    print("Accuracy =", accuracy_score(y_test_original,predictions))
    precision, recall, _ = precision_recall_curve(y_test_original, predictions)
    p1.line(precision, recall, line_width=2,line_alpha=0.6,name ="line2")
    average_precision = average_precision_score(y_test_original, predictions)
    p1.title.text = "Average Precision Score %f" % average_precision
    #print(np.unique(predictions))
    tn, fp, fn, tp = confusion_matrix(y_test_original,predictions).ravel()
    source.data =dict(fruits=fruits, counts=[tp, fp, tn, fn])
    p.title.text = "Model Accuracy %f" % accuracy_score(y_test_original,predictions)

controls = [features, warm_start, copy_X_train]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

sizing_mode = 'fixed'  # 'scale_width' also looks nice with this example

inputs = widgetbox(*controls, sizing_mode=sizing_mode)
l = layout([
    [desc,stats],
    [inputs,tabs]
])

#update()  # initial load of the data
 
curdoc().add_root(l)
curdoc().title = "Churn" 