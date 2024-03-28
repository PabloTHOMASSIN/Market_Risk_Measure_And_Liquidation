import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Preprocessing 
data= pd.read_csv(r"Natixis stock (dataset TD12).txt", delimiter='\t', header=None, parse_dates=[0]) #We will use mainly dataframe to analyze data in our whole project
data.columns = ['Date', 'Value'] # We will tend to always rename columns
data['Date'] = pd.to_datetime(data['Date'], format="%d/%m/%Y") # Preprocessing of date-time
data['Value'] = data['Value'].str.replace(',', '.').astype(float) #Re type for float number as we read a txt, the  coma - point transition isn't immediate
data = data.sort_values('Date') # For now let's sort by date to calculate returns coherent we use build in sort algorithm.

#Compute of Returns in data frame and creating new column, the shift() function just go to the next row in the same column, we didn't use pct_change() a builded in function of pandas
def Returns(data):
    data['Returns'] = (data['Value'] - data['Value'].shift(1)) / data['Value'].shift(1)
    return data

#Let's compute returns based on sorted date (which is the good sort)
data = Returns(data)
print(data)
#As we can see the first value of returns is NaN which is logical due to the definition of returns so we decide to erase the row from our dataset
data = data.dropna()
print(data)

# Let's define some value alpha is the probability level targeted and value portfolio is the actual value of the portfolio
alpha = 0.05
value_portfolio = 100000

#Now on our still data-sorted dataset we will select only the date we wanted from the exercise for question 1 and 2
data_date_sort_2015 = data[(data["Date"] >= '2015-01-01') & (data["Date"] < '2016-12-31')]
data_date_sort_2017 = data[(data["Date"] >= '2017-01-01') & (data["Date"] < '2018-12-31')]
#Once the selection is done now we can sort returns ! 
data_date_sort_2015 = data_date_sort_2015.sort_values("Returns") # using build in sort function of panda
data_date_sort_2017 = data_date_sort_2017.sort_values("Returns") # same as ligne 32
print(data_date_sort_2015) # quick check up

#Let's define kernel with regards to exercise and with respect to the indicatrice
def kernel(u):
    if abs(u)<=1 :
        return (15/16)*((1-(u)**2)**2)  
    else:
        return 0
    
#Let's define our kernel density
def kernel_density(data,x):  # data is the whole column of returns and x is a particular return
    h= ((4 * data['Returns'].std())/(3 * len(data['Returns'])))**0.2 # h with respect to the thumb formula

    n=len(data['Returns'])
    somme_kernel_density=0
    
    for datas in data['Returns']:
        somme_kernel_density += kernel((x-datas)/h) # we sum based on our kernel function given precedently
                                        
    return (somme_kernel_density/(n*h)) # density function obtained mathematically

    
def VaR_Kernel(data, alpha):
    probabilities = [] # we want to create an array of all our probability

    #Let's evaluate every probability for each returns
    for datas in data['Returns']: 
        probability = kernel_density(data, datas)
        probabilities.append(probability)

    total = sum(probabilities) # we sum all the probabilities cause we have to secure a condtion of the sum (probabilities ) == 1

    # Normalization in order to have the condition spoken about in the upper line
    if total != 0:
        probabilities = [prob / total for prob in probabilities]

    #Let's compute the quantile of distribution
    sorted_returns = sorted(data['Returns'])
    cumulative_prob = 0

    for i, prob in enumerate(probabilities):#Here we use enumerate to make each probabilities an indenpendent object  as such there is a new indexation which we can track with i
        cumulative_prob += prob
        if cumulative_prob >= alpha:
            resultat = sorted_returns[i]
            break

    return resultat

#Now let's compute our VaR with alpha = 0.05 on our data date selected, sorted on returns
Value_at_Risk = VaR_Kernel(data_date_sort_2015,0.05)
print(f"Value at Risk of Kernel : {Value_at_Risk} \n") # let's Print our result

#Let's compute our over thresholds
def Over_VaR(data, val_ref):
    somme=0 # initialization of the sum
    
    for i in data['Returns']: # Let's count all value of returns under the threshold
        if (i<val_ref):
            somme+=1
            
    proportion= (somme/len(data['Returns']))*100 # Let's compute the percentage
    
    return proportion

Over_Threshold_Returns = Over_VaR(data_date_sort_2017, Value_at_Risk)
print(f"Percentage over threshold for a VaR {Over_Threshold_Returns}")


def ES_nonparam_var(data, Value_at_Risk):
    
    over_threshold = data["Returns"][data["Returns"] < Value_at_Risk]
    ESnonparam=np.mean(over_threshold)
    return ESnonparam


ES_varnonparam = ES_nonparam_var(data_date_sort_2017,Value_at_Risk)
print(f"Expected Shortfall vaR non-paramétrique: {ES_varnonparam} \n")

