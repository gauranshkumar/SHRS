# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import smtplib
from IPython.display import display
from email.message import EmailMessage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
class Prediction_system:
    def __init__(self, path):
        self.path = path
        self.Data = pd.read_csv(path)
        display(self.Data.head())
        self.Min = np.min(self.Data['sugar'])
        self.Max = np.max(self.Data['sugar'])

    def fit(self):
        """Data: Pandas Dataframe"""
        self.normalize()
        display(self.Data.head())
        y = self.Data['label']
        x = self.Data.drop(["label"], axis=1).to_numpy()
        xTrain, yTrain = x[:80], y[:80]
        xTest, yTest = x[80:], y[80:]
        self.clf = GradientBoostingClassifier(
            n_estimators=100, learning_rate=.01, max_depth=1, random_state=0)
        self.clf.fit(xTrain, yTrain)
        print("Model Trained\nAccuracy : {}".format(
            self.clf.score(xTest, yTest)))

    def single(self, num):
        return num

    def normalize(self, single=0):
        """Data : Dataframe"""
        scaler = MinMaxScaler()
        x = np.reshape(self.Data['sugar'].to_numpy(), (-1, 1))
        scaler = scaler.fit(x)
        if(single == 0):
            sugar_normalised = scaler.transform(x)
            self.Data['sugar'] = sugar_normalised
            print("Executed Zero")
        else:
            print(single, self.Min, self.Max)
            return np.array((single - self.Min) / (self.Max - self.Min))

    def predict(self, sugar, insulin, exercise, diet,  severe_disease):
        return self.clf.predict(np.reshape([self.normalize(sugar), insulin, exercise, diet, severe_disease], (1, -1)))


# %%
# shrs = SHRS_PS("./data_raka.csv")


# %%
# shrs.fit()


# %%
# shrs.predict(299, 1, 1, 1, 1)


# %%
class Allocation_system:
    def __init__(self, data):
        """data: Dictionary of user info"""
        self.user = data
        self.predictedSiverity = 0
        self.automatedSiverity = ""
        self.predicteAllocation = ""

    def result(self):
        """return: String - Final Report String"""
        # params = []
        # params.append(float(user.data['sugar']))
        # params.append(float(user.data['insulin']))
        # params.append(float(user.data['exercise']))
        # params.append(float(user.data['diet']))
        # params.append(float(user.data['severe_disease']))
        predictionSystem = Prediction_system("./data_raka.csv")
        predictionSystem.fit()
        self.predictedSiverity = predictionSystem.predict(float(self.user['sugar']), float(self.user['insulin']), float(
            self.user['exercise']), float(self.user['diet']), float(self.user['severe_disease']))[0]
        return self.predictedSiverity

    def report(self):
        print(self.result())
        if(self.predictedSiverity == 1):
            self.automatedSiverity = "1 - Low Chances of Re-addmitance"
        elif(self.predictedSiverity == 2):
            self.automatedSiverity = "2 - Medium Chances of Re-addmitance"
        elif(self.predictedSiverity == 3):
            self.automatedSiverity = "3 - High Chances of Re-addmitance"

        if(((self.user['age'] > 60) and (self.predictedSiverity >= 2)) and (self.user["cardiac"] == 1)):
            self.predicteAllocation = '''Ward Type : ICCU (Intensive Cardiac Care Unit)\n
            Ward Boys/ Nurses : 2-3\n
            Stretcher/Weelchair : Yes\n
            Specialised Doctor : Yes\n'''

        elif((self.user['age'] > 60) and (self.predictedSiverity >= 2)):
            self.predicteAllocation = '''Ward Type : ICU (Intensive Care Unit)\n
            Ward Boys/ Nurses : 2-3\n
            Stretcher/Weelchair : Yes\n
            Specialised Doctor : Yes\n'''

        elif((self.user['age'] <= 60) and (self.predictedSiverity >= 2)):
            self.predicteAllocation = '''Ward Type : ICU (Intensive Care Unit)\n
            Ward Boys/ Nurses : 2-3\n
            Stretcher/Weelchair : No\n
            Specialised Doctor : Yes\n'''

        elif((self.user['age'] <= 60) and (self.predictedSiverity < 2)):
            self.predicteAllocation = '''\n\tWard Type : ICCU (Intensive Cardiac Care Unit)\n\t
            Ward Boys/ Nurses : 1-2\n\t
            Stretcher/Weelchair : No\n\t
            Specialised Doctor : No\n'''

        r = """
        Name : {}\n
        Gender : {}\n
        Age : {}\n
        Random Sugar Level : {} mg/dL\n
        Insulin Dosage : {}\n
        Exercise : {}\n
        Diet : {}\n
        Severe Disease : {}\n
        Calculated Criticalness: {}\n
        Predicted Allocation : {}\n
        """.format(self.user['name'], self.user['gender'], self.user['age'], self.user['sugar'], self.user['insulin'], self.user['exercise'], self.user['diet'], self.user['severe_disease'], self.automatedSiverity, self.predicteAllocation)
        return r

    def verification(self):
        msg = EmailMessage()
        msg.set_content(self.report())

        msg['Subject'] = "Patient Report for Verification"
        msg['From'] = "temp.mail.saga@gmail.com"
        msg['To'] = "satyammishrawe@gmail.com"

        # Send the message via our own SMTP server.
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login("temp.mail.saga@gmail.com", "foobar#2001")
        server.send_message(msg)
        server.quit()

    def submit(self, is_critical=False):
        if(is_critical):
            self.verification()
        elif(self.predictedSiverity >= 2):
            self.verification()
        return self.report()

# %%
