from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
import os
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,
confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import joblib

main = Tk()
main.title(&quot;Social Media Trend Analyzer&quot;)
main.geometry(&quot;1300x1200&quot;)
global filename
global X, Y
global tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test
global tfidf_vectorizer
global accuracy,error
stop_words = set(stopwords.words(&#39;english&#39;))
lemmatizer = WordNetLemmatizer()
global X_train_lstm_features, X_test_lstm_features
def cleanPost(doc):
tokens = doc.split()
table = str.maketrans(&#39;&#39;, &#39;&#39;, punctuation)
tokens = [w.translate(table) for w in tokens]
tokens = [word for word in tokens if word.isalpha()]
tokens = [w for w in tokens if not w in stop_words]
tokens = [word for word in tokens if len(word) &gt; 1]
tokens = [lemmatizer.lemmatize(token) for token in tokens]
tokens = &#39; &#39;.join(tokens)
return tokens
def uploadDataset():
global filename,dataset
text.delete(&#39;1.0&#39;, END)
filename = filedialog.askopenfilename(initialdir=&quot;TwitterNewsData&quot;)
dataset = pd.read_csv(filename)
text.insert(END,str(dataset.head())+&quot;\n&quot;)
def preprocess():
global X_train, X_test, y_train, y_test
textdata = []
labels = []
text.delete(&#39;1.0&#39;, END)



for i in range(len(dataset)):
msg = dataset.at[i, &#39;text&#39;]
label = dataset.at[i, &#39;target&#39;]
msg = str(msg)
msg = msg.strip().lower()
labels.append(int(label))
clean = cleanPost(msg)
textdata.append(clean)
tokenizer = Tokenizer(num_words=10000) # Adjust as needed
tokenizer.fit_on_texts(textdata)
with open(&#39;tokenizer.pickle&#39;, &#39;wb&#39;) as handle:
pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
sequences = tokenizer.texts_to_sequences(textdata)
X = pad_sequences(sequences, maxlen=100) # Adjust maxlen as needed
Y = labels
le = LabelEncoder()
Y = le.fit_transform(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
text.insert(END,&quot;\n\nTotal Tweet found in dataset : &quot;+str(len(X))+&quot;\n&quot;)
text.insert(END,&quot;Total records used to train algorithms : &quot;+str(len(X_train))+&quot;\n&quot;)
text.insert(END,&quot;Total records used to test algorithms : &quot;+str(len(X_test))+&quot;\n&quot;)
return X_train, X_test, y_train, y_test
def xgboost():
global X_train, X_test, y_train, y_test
model_folder = &#39;model&#39;
model_path = os.path.join(model_folder, &#39;xgb.pkl&#39;)
if os.path.exists(model_path):
clf = joblib.load(model_path)
y_pred = clf.predict(X_test)
else:
clf = XGBClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
joblib.dump(clf, model_path)
print(&quot;XGBClassifier model trained and model weights saved.&quot;)


a = accuracy_score(y_test, y_pred)*100
print(f&#39;Accuracy: {a}&#39;)
# Precision, Recall, F1-score
p = precision_score(y_test, y_pred, average=&#39;weighted&#39;)*100
r = recall_score(y_test, y_pred, average=&#39;weighted&#39;)*100
f = f1_score(y_test, y_pred, average=&#39;weighted&#39;)*100
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(&#39;Confusion Matrix:&#39;)
print(cm)
# Classification Report
cr = classification_report(y_test, y_pred)
print(&#39;Classification Report:&#39;)
print(cr)
text.insert(END, &quot;Precision : &quot; + str(p) + &quot;\n&quot;)
text.insert(END, &quot;Recall : &quot; + str(r) + &quot;\n&quot;)
text.insert(END, &quot;FMeasure : &quot; + str(f) + &quot;\n&quot;)
text.insert(END, &quot;Accuracy : &quot; + str(a) + &quot;\n\n&quot;)
# Heatmap for Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=&#39;d&#39;, cmap=&#39;Blues&#39;, xticklabels=[&#39;Class 0&#39;, &#39;Class 1&#39;],
yticklabels=[&#39;Class 0&#39;, &#39;Class 1&#39;])
plt.xlabel(&#39;Predicted&#39;)
plt.ylabel(&#39;Actual&#39;)
plt.title(&#39;XGBoost Confusion Matrix Heatmap&#39;)
plt.show()
# Modify your create_lstm_model function
def create_lstm_model(input_dim, output_dim, input_length):
model = Sequential()
model.add(Embedding(input_dim=input_dim, output_dim=output_dim,
input_length=input_length))
model.add(Bidirectional(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(64, activation=&#39;relu&#39;))
model.add(Dropout(0.5))
model.add(Dense(1, activation=&#39;sigmoid&#39;))
return model
def extract_lstm_features(model, X_data):


# Extract features using the LSTM model
intermediate_layer_model = Sequential(model.layers[:-1]) # Exclude the output layer
features = intermediate_layer_model.predict(X_data)
return features
def train_random_forest(X_train_features, y_train):
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train_features, y_train)
return rf_classifier
def evaluate_model(classifier, X_test_features, y_test):
y_pred = classifier.predict(X_test_features)
text.delete(&#39;1.0&#39;, END)

# Accuracy
a = accuracy_score(y_test, y_pred)*100
print(f&#39;Accuracy: {a}&#39;)

# Precision, Recall, F1-score
p = precision_score(y_test, y_pred, average=&#39;weighted&#39;)*100
r = recall_score(y_test, y_pred, average=&#39;weighted&#39;)*100
f = f1_score(y_test, y_pred, average=&#39;weighted&#39;)*100

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(&#39;Confusion Matrix:&#39;)
print(cm)

# Classification Report
cr = classification_report(y_test, y_pred)
print(&#39;Classification Report:&#39;)
print(cr)
text.insert(END, &quot;Precision : &quot; + str(p) + &quot;\n&quot;)
text.insert(END, &quot;Recall : &quot; + str(r) + &quot;\n&quot;)
text.insert(END, &quot;FMeasure : &quot; + str(f) + &quot;\n&quot;)
text.insert(END, &quot;Accuracy : &quot; + str(a) + &quot;\n\n&quot;)

# Heatmap for Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=&#39;d&#39;, cmap=&#39;Blues&#39;, xticklabels=[&#39;Class 0&#39;, &#39;Class 1&#39;],
yticklabels=[&#39;Class 0&#39;, &#39;Class 1&#39;])
plt.xlabel(&#39;Predicted&#39;)
plt.ylabel(&#39;Actual&#39;)
plt.title(&#39;LSTM Confusion Matrix Heatmap&#39;)
plt.show()
def runLSTM():
global rf_model
if os.path.exists(&#39;lstm_model.h5&#39;) and os.path.exists(&#39;random_forest_model.joblib&#39;):
lstm_model = load_model(&#39;lstm_model.h5&#39;)
rf_model = joblib.load(&#39;random_forest_model.joblib&#39;)
X_train_lstm_features = extract_lstm_features(lstm_model, X_train)
X_test_lstm_features = extract_lstm_features(lstm_model, X_test)
else:
# Create and train LSTM model
lstm_model = create_lstm_model(input_dim=10000, output_dim=50,
input_length=100)
lstm_model.compile(optimizer=&#39;adam&#39;, loss=&#39;binary_crossentropy&#39;,
metrics=[&#39;accuracy&#39;])
lstm_model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

# Extract LSTM features
X_train_lstm_features = extract_lstm_features(lstm_model, X_train)
X_test_lstm_features = extract_lstm_features(lstm_model, X_test)
# Train Random Forest on LSTM features
rf_model = train_random_forest(X_train_lstm_features, y_train)


# Save LSTM model
lstm_model.save(&#39;lstm_model.h5&#39;)
# Save Random Forest model
joblib.dump(rf_model, &#39;random_forest_model.joblib&#39;)
# Evaluate the Random Forest model
evaluate_model(rf_model, X_test_lstm_features, y_test)
def predict():
testfile = filedialog.askopenfilename(initialdir=&quot;TwitterNewsData&quot;)
testData = pd.read_csv(testfile)
text.delete(&#39;1.0&#39;, END)
# Load LSTM model
lstm_model = load_model(&#39;lstm_model.h5&#39;)

# Load Tokenizer
with open(&#39;tokenizer.pickle&#39;, &#39;rb&#39;) as handle:
tokenizer = pickle.load(handle)
testData = testData.values
testData = testData[:, 0]
print(testData)
for i in range(len(testData)):
msg = testData[i]
msg1 = testData[i]
print(msg)
review = msg.lower()
review = review.strip().lower()
review = cleanPost(review)

# Tokenize and pad the new test data for LSTM model
sequence = tokenizer.texts_to_sequences([review])
padded_sequence = pad_sequences(sequence, maxlen=100)
# Extract LSTM features for the new test data
lstm_features = extract_lstm_features(lstm_model, padded_sequence)

# Predict using Random Forest model
predict_rf = rf_model.predict(lstm_features)



# Display the results
if predict_rf == 0:
text.insert(END, msg1 + &quot; === Given Tweet predicted as TREND\n\n&quot;)
else:
text.insert(END, msg1 + &quot; == Given Tweet predicted as NO TREND\n\n&quot;)
font = (&#39;times&#39;, 15, &#39;bold&#39;)
title = Label(main, text=&#39;Deep Learning-based Social Media Trend Analyzer to Predict the
Trends Over Time&#39;)
title.config(bg=&#39;medium purple&#39;, fg=&#39;thistle1&#39;)
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0,y=5)
font1 = (&#39;times&#39;, 13, &#39;bold&#39;)
ff = (&#39;times&#39;, 12, &#39;bold&#39;)
uploadButton = Button(main, text=&quot;Upload Trend Tweet Dataset&quot;,
command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)
processButton = Button(main, text=&quot;Preprocess Dataset&quot;, command=preprocess)
processButton.place(x=20,y=150)
processButton.config(font=ff)
dtButton = Button(main, text=&quot;Run LSTM Algorithm&quot;, command=runLSTM)
dtButton.place(x=20,y=200)
dtButton.config(font=ff)
dtButton = Button(main, text=&quot;Existing XGB Algorithm&quot;, command=xgboost)
dtButton.place(x=20,y=300)
dtButton.config(font=ff)
predictButton = Button(main, text=&quot;Test Tweet Detection&quot;, command=predict)
predictButton.place(x=20,y=250)
predictButton.config(font=ff)
font1 = (&#39;times&#39;, 12, &#39;bold&#39;)
text=Text(main,height=30,width=100)
scroll=Scrollbar(text)


text.configure(yscrollcommand=scroll.set)
text.place(x=330,y=100)
text.config(font=font1)
main.config(bg=&#39;lavender&#39;)
main.mainloop()
