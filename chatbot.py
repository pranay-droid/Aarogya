import bart
import pandas as pd
classifier=bart.classifier
l=[]

print("What is your body temperature?")
sequence_to_classify = input()
candidate_labels = ['1', '0']
ans=classifier("Do I have a fever if my body temperature is (given that anything greater than 99 degree farenheit is fever),"+sequence_to_classify, candidate_labels)
l.append(ans['labels'][0])
print("Have you experienced any coughing recently?")
sequence_to_classify = input()
candidate_labels = ['1', '0']
ans=classifier(sequence_to_classify, candidate_labels)
l.append(ans['labels'][0])
print("Do you feel abnormally tired? / Have you observed any signs of fatigue?")
sequence_to_classify = input()
candidate_labels = ['1', '0']
ans=classifier(sequence_to_classify, candidate_labels)
l.append(ans['labels'][0])
print("Are you facing any difficulties in breathing?")
sequence_to_classify = input()
candidate_labels = ['1', '0']
ans=classifier(sequence_to_classify, candidate_labels)
l.append(ans['labels'][0])
print("What's your age?")
x=input()
l.append(x)

print("What is your sexual orientation?")
sequence_to_classify = input()
candidate_labels = ['1', '0']
ans=classifier(sequence_to_classify, candidate_labels)
l.append(ans['labels'][0])

print("What is your last measured blood pressure?")
sequence_to_classify = input()
candidate_labels = ['Low','High','Medium']
ans=classifier("my blood pressure , the value is"+sequence_to_classify, candidate_labels)
l.append(ans['labels'][0])
print("What is your cholesterol level?")
sequence_to_classify = input()
candidate_labels = ['High','Medium','Low']
ans=classifier("less than 200 is "+sequence_to_classify, candidate_labels)
l.append(ans['labels'][0])

df = pd.DataFrame(l)
df.columns=['']
df.index=['Fever','Cough','Fatigue','Difficulty Breathing','Age','Gender','Blood Pressure','Cholesterol Level']
df=df.transpose()
#df.drop(df.columns[[0]], axis = 1, inplace = True)
#d = df.pivot(index='ID', columns='Identifier', values='Value')
df.to_csv("Out.csv")