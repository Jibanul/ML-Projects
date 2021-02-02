# Author: Md Jibanul Haque Jiban
# Homework 1
# CAP 5610: Machine Learning

# required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.stats import chi2_contingency

# import data
train_df = pd.read_csv('/Users/mdjibanulhaquejiban/PhD_CRCV/Semesters/Spring2021/ML/HW/HW1/Titanic/train.csv')
test_df = pd.read_csv('/Users/mdjibanulhaquejiban/PhD_CRCV/Semesters/Spring2021/ML/HW/HW1/Titanic/test.csv')
combine = [train_df, test_df]

#######################

## Q1
print(train_df)
print(train_df.describe())
print(train_df.info())

## Q2-Q4
print(train_df.info())

## Q5
print(train_df.info())
print(test_df.info())

## Q6
print(train_df.head())


## Q7
# create a sub-dataframe with only numerical features
train_df_num = train_df[['Age', 'SibSp', 'Parch', 'Fare']]
print(train_df_num.describe())

## Q8
# create a sub-dataframe with only categorical features
train_df_cat = train_df[['Survived', 'Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'PassengerId']]
train_df_cat = train_df_cat.astype('object')
# print(train_df_cat.info())
train_df_cat.describe(include=[object])


## Q9
contigency_table= pd.crosstab(train_df['Pclass'], train_df['Survived']) 
print('The contigency table:')
print('\n')
print(contigency_table)

# Chi-Sq test
chi2, p_value, deg_freedom, expected = chi2_contingency(contigency_table) 

print('\n')
print('The test statistic is', chi2)
print('\n')
print('The p-value of the test is', p_value)
print('\n')
print('Degrees of freedom is', deg_freedom)
print('\n')
print('The expected frequencies, based on the marginal sums of the table. \n', expected)



## Q10
female = np.where((train_df['Sex']=='female'))
female_survived = np.where((train_df['Sex']=='female') & (train_df['Survived'] == 1))
print("The ratio of female survivals in training set is", len(female_survived[0])/len(female[0]))

## Chi-Sq
contigency_table= pd.crosstab(train_df['Sex'], train_df['Survived']) 
print('The contigency table:')
print('\n')
print(contigency_table)

# Chi-Sq test
chi2, p_value, deg_freedom, expected = chi2_contingency(contigency_table) 

print('\n')
print('The test statistic is', chi2)
print('\n')
print('The p-value of the test is', p_value)



## Q11
survived_age = train_df.loc[np.where((train_df['Survived'] == 1))]['Age']
not_survived_age = train_df.loc[np.where((train_df['Survived'] == 0))]['Age']

# survived histogram
survived_age.hist(bins=21, color='orange')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age of People who survived (= 1)')
plt.show()
plt.close()

# not survived histogram
not_survived_age.hist(bins=21, color='tomato')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age of People who did not survive (= 0)')
plt.show()
plt.close()


## Q12
# Create data
not_survived_pclass1_age = train_df.loc[np.where((train_df['Pclass'] == 1) & (train_df['Survived'] == 0))]['Age']
not_survived_pclass2_age = train_df.loc[np.where((train_df['Pclass'] == 2) & (train_df['Survived'] == 0))]['Age']
not_survived_pclass3_age = train_df.loc[np.where((train_df['Pclass'] == 3) & (train_df['Survived'] == 0))]['Age']

survived_pclass1_age = train_df.loc[np.where((train_df['Pclass'] == 1) & (train_df['Survived'] == 1))]['Age']
survived_pclass2_age = train_df.loc[np.where((train_df['Pclass'] == 2) & (train_df['Survived'] == 1))]['Age']
survived_pclass3_age = train_df.loc[np.where((train_df['Pclass'] == 3) & (train_df['Survived'] == 1))]['Age']

# plot figures
fig, axs = plt.subplots(3,2,figsize=(12,12))
fig.suptitle('Distributions of Age by Pclass and Survived')
axs[0,0].hist(not_survived_pclass1_age, bins=21, color='tomato')
axs[0,0].set_title('Pclass = 1 | Survived = 0')
axs[1,0].hist(not_survived_pclass2_age, bins=21, color='tomato')
axs[1,0].set_title('Pclass = 2 | Survived = 0')
axs[2,0].hist(not_survived_pclass3_age, bins=21, color='tomato')
axs[2,0].set_title('Pclass = 3 | Survived = 0')

axs[0,1].hist(survived_pclass1_age, bins=21, color='orange')
axs[0,1].set_title('Pclass = 1 | Survived = 1')
axs[1,1].hist(survived_pclass2_age, bins=21, color='orange')
axs[1,1].set_title('Pclass = 2 | Survived = 1')
axs[2,1].hist(survived_pclass3_age, bins=21, color='orange')
axs[2,1].set_title('Pclass = 3 | Survived = 1')
plt.show()
plt.close()


# Count number of passengers by pclass
train_df.groupby(['Pclass'])['PassengerId'].count()
train_df.groupby(['Pclass', 'Survived'])['PassengerId'].count()



## Q13
train_df_q13 = train_df.groupby(['Embarked', 'Survived', 'Sex'])['Fare'].mean()

# plot figures
fig, axs = plt.subplots(3,2,figsize=(11,11))
fig.suptitle('Distributions of Average Fare by Embarked, Survived and Sex')

axs[0,0].bar(['female', 'male'],train_df_q13[8:10].values, color='tomato')
axs[0,0].set_title('Embarked = S | Survived = 0')
axs[0,0].set_ylabel('Average Fare')
axs[1,0].bar(['female', 'male'],train_df_q13[:2].values, color='tomato')
axs[1,0].set_title('Embarked = C | Survived = 0')
axs[1,0].set_ylabel('Average Fare')
axs[2,0].bar(['female', 'male'],train_df_q13[4:6].values, color='tomato')
axs[2,0].set_title('Embarked = Q | Survived = 0')
axs[2,0].set_ylabel('Average Fare')


axs[0,1].bar(['female', 'male'],train_df_q13[10:12].values, color='orange')
axs[0,1].set_title('Embarked = S | Survived = 1')
axs[1,1].bar(['female', 'male'],train_df_q13[2:4].values, color='orange')
axs[1,1].set_title('Embarked = C | Survived = 1')
axs[2,1].bar(['female', 'male'],train_df_q13[6:8].values, color='orange')
axs[2,1].set_title('Embarked = Q | Survived = 1')
plt.show()
plt.close()


train_df.groupby(['Embarked', 'Survived', 'Sex'])['Fare'].mean()
train_df.groupby(['Embarked', 'Survived', 'Sex'])['PassengerId'].count()



## Q14
train_df.Ticket.duplicated().value_counts()


## Q15
train_df.Cabin.describe()
test_df.Cabin.describe()



## Q16
train_df['Gender'] = np.where(train_df['Sex']== 'male', 0, 1)
train_df.head(10)


## Q17
# calculate mean and standard deviation
mean = train_df['Age'].mean()
std = train_df['Age'].std()

print('Mean', mean)
print('Standard Deviation', std)
print('\n')
print('Estimated Missing Values in the Age feature.')
# we can randomly pick a value between standard deviation and Mean from Uniform distribution
# to impute missing values
def missing_value_imputation(value):
    if np.isnan(value) == True: 
        value = random.uniform(std, mean)
    else:
         value = value
    return value

# call the above function
train_df['Age'] = train_df['Age'].apply(missing_value_imputation) 


## Q18
# find the most frequent value
most_frequent_value = train_df['Embarked'].value_counts().idxmax()
print('The most frequent value in Embarked:', most_frequent_value)
print('\n')


print('The training set with missing Embarked records')
is_na = train_df["Embarked"].isna()
print(train_df[is_na]["Embarked"])

# fill the missing records by the most frequent value
train_df["Embarked"] = train_df["Embarked"].fillna(most_frequent_value)

print('\n')
print('The training set without missing Embarked records')
print(train_df[is_na]["Embarked"])



## Q19
# find the most frequent value
mode = test_df['Fare'].mode()
print('The most frequent value in Fare:', mode)
print('\n')


print('The test set with missing Fare records')
is_na = test_df["Fare"].isna()
print(test_df[is_na]["Fare"])

# fill the missing records by the most frequent value
test_df["Fare"] = test_df["Fare"].fillna(mode[0])

print('\n')
print('The test set without missing Fare records')
print(test_df[is_na]["Fare"])




## Q20
train_df['ordinal_fare'] = np.where(train_df['Fare'] <= 7.91, 0, 
                (np.where(train_df['Fare'] <= 14.454, 1, 
                    (np.where(train_df['Fare'] <= 31.0, 2, 3)))))

# print first 10 rows
# print(train_df.head(10))
train_df[['PassengerId','Fare','ordinal_fare']].head(10)

# reproduce the table in the question
Avg = pd.DataFrame(train_df.groupby(['ordinal_fare'])['Survived'].mean())
Avg = pd.DataFrame(Avg)
Avg



#### The end ####