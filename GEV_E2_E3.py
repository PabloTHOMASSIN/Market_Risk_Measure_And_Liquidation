import numpy as np
import pandas as pd
#Preprocessing 
data= pd.read_csv(r"Natixis stock (dataset TD12).txt", delimiter='\t', header=None, parse_dates=[0]) #We will use mainly dataframe to analyze data in our whole project
data.columns = ['Date', 'Value'] # We will tend to always rename columns
data['Date'] = pd.to_datetime(data['Date'], format="%d/%m/%Y") # Preprocessing of date-time
data['Value'] = data['Value'].str.replace(',', '.').astype(float) #Re type for float number as we read a txt, the  coma - point transition isn't immediate
data = data.sort_values('Date') # For now let's sort by date to calculate returns coherent we use build in sort algorithm.
alpha = 0.05 # Definition of a confidence level for the begining of the program

#Compute of Returns in data frame and creating new column, the shift() function just go to the next row in the same column, we didn't use pct_change() a builded in function of pandas
def Returns(data):
    data['Returns'] = (data['Value'] - data['Value'].shift(1)) / data['Value'].shift(1)
    return data
#Let's compute returns based on sorted date (which is the good sort)
data = Returns(data)
print(data)
#Now let's drop our NaN Value
data = data.dropna()
print(data) 

#Let's define of a new column, profits with usage of the apply() methods of panda to apply selecting condition
def Profit(data):
    data['Profits'] = data['Returns'].apply(lambda x: x if x > 0 else  0)
    return data
data = Profit(data)
data = data.dropna()
data= data.sort_values("Profits")
#Print data cleaned and sorted for profits
print(data)

#Let's now do the same for Losses
def Loss(data):
    data['Losses'] = data['Returns'].apply(lambda x: x if x < 0 else 0)
    return data
data = Loss(data)
data = data.dropna()
data= data.sort_values("Losses")
#And we can print our result
print(data)

#This function compute the quantile of level 1 - alpha
def Quantile(series, p):
    sorted_series = series.sort_values(ascending=True) #let's sort the given series in case
    
    if 0 <= p <= 1:  # if p >=0 and p <= 1 then we can use the index computing as follow
        index = int(p * len(sorted_series))
        return sorted_series.iloc[index]
    else : 
        if p < 0: #then the minimum we can go for the index is 0, in this cas we can do the same for p > 1 but was not needed further in the code
            return sorted_series.iloc[0]
        if p > 1 :
            return sorted_series.iloc[len(sorted_series)]

#Let's define pickands estimator with respect to formula of the course page 191
def Pickands_Estimator(data, alpha):
    n = len(data)
    
    k = Quantile(data, 1 - alpha)
    k_2n = Quantile(data, 1 - 2 * alpha)
    k_4n = Quantile(data, 1 - 4 * alpha)

    
    
    a = np.log(k - k_2n)
    b = 0
    #Let's define a security condition to make sure b is not zero and so we don't divide by zero
    if k_2n != k_4n  : 
        b = np.log(k_2n - k_4n)
    else:
        b = np.nan
        
    c = a - b

    return 1 / np.log(2) * c

# Selection profits and losses with value different of zero
profits = data[data['Profits'] != 0]['Profits']
losses = data[data['Losses'] != 0]['Losses']
# Estimate Pickands for profits
pickands_profits = Pickands_Estimator(profits, alpha) 
# Estimate Pickands for losses
pickands_losses = Pickands_Estimator(losses, alpha)  
#Printing our result
print(f"Pickands estimator for profits:{pickands_profits}")
print(f"Pickands estimator for losses:{pickands_losses}")

#Let's define many confidence level
alpha = np.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50, 0.75, 0.95])
#Let's select all the returns different of zero
returns = data[data['Returns'] !=0]['Returns']


#Let's now define the VaR for pickands estimator with respect to the course on slide 198
def VaR_Pickands(data,alpha):
    n=len(data)
    k =  Quantile(data, 1 - alpha)
    k_2n = Quantile(data, 1 - 2 * alpha)
    numerator = (1 /n*(1-alpha))**(Pickands_Estimator(data, alpha)) - 1 #with respect to the course page 197 we can deduce k = 1 is also true
    denominator= 1 - 2 **(-Pickands_Estimator(data,alpha))
    #If the denominator is 0 which can happen if the pickand estimator is 0 can happen if the confidence level is too high
    if denominator !=0 : 
        return (numerator/denominator)*(k - k_2n) + k
    else :
        return np.nan

# Compute VaR for returns using Pickands estimator for different alpha levels
var_returns_pickands = np.array([VaR_Pickands(returns, a) for a in alpha])
print(f"VaR using Pickands estimator for returns:{var_returns_pickands}")
    