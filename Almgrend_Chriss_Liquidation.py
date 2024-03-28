import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Load data from Excel file & Process the column of it
data = pd.read_excel(r"C:\Users\pablo\OneDrive - De Vinci\COURS\A4\S1\Market Risk\TD\TD_4\Dataset_TD4.xlsx")
data.columns = ['Date', 'Bid-Ask spread', 'Volume of transaction', 'Sign of the transaction', 'Price', '']
data = data.drop('', axis=1) # My local dataset may have been corrupted so I had to discard an empty column
data_copy = data # This is a copy of the data set for later usage.


# X & T Parameters 
X_liquidate = 1E6
T = len(data)
# Here we define a return computing function, the shift() function is used to go down of one row in the same column, otherwise we could use .pct_change()
def Returns(data):
    data['Returns'] = (data['Price'].shift(1) - data['Price']) / data['Price']
    return data
data = Returns(data) # Adding a Returns column to our data set
print(data) # printing our data set 

#Now we express our returns as a numpy array vector ignoring our first row
data_returns = data['Returns'].values.astype(float)[1:]

#Using an expression in numpy array we compute our first parameters
average_spread = np.mean(data['Bid-Ask spread'].values.astype(float))
sigma = np.std(data_returns) 
epsilon = average_spread / 2
tau = 1 / 24.0
lam = 2E-6  # This parameters has been arbitrarily choosed based on the course

#Now we introduce ou substration methods for price remembering the fact that it's price before transaction
def Delta_Price_gamma(data): 
    data['Delta_Price_Gamma'] =  - data['Price'] + data['Price'].shift(-1) 
    return data
#And according to the eta parameters of price we need to do the oposit
def Delta_Price_eta(data): 
    data['Delta_Price_Eta'] =   data['Price'] - data['Price'].shift(-1) 
    return data
# We apply our new function to our dataset
data = Delta_Price_gamma(data)
data = Delta_Price_eta(data)
print(data)


# Let's now clean our data set from it's NaN value in order to prepare the linear regression
data = data.dropna(subset=['Delta_Price_Gamma','Delta_Price_Eta','Volume of transaction'])
print(data)


# Now let's express our variable in the form of numpy array 
data_delta_price_gamma = data['Delta_Price_Gamma'].values.astype(float)
data_delta_price_eta = data['Delta_Price_Eta'].values.astype(float)
# For volume we have (in order to get linear regression) to express the volume with it's sign so we make the product of two numpy array (same for the volume squarred)
volume_signed = (data['Volume of transaction'].values.astype(int)) * (data['Sign of the transaction'].values.astype(int))
volume_signed_squared = (data['Volume of transaction'].values.astype(int))*(data['Volume of transaction'].values.astype(int)) *(data['Sign of the transaction'].values.astype(int))

# Linear regression parameter definition for gamma
X_gamma = volume_signed.reshape(-1,1) # here we are only looking for impact of order 1
y_gamma = data_delta_price_gamma

# Model's fitting
model = LinearRegression()
model.fit(X_gamma, y_gamma)

# Extraction of "coefficient directeur"
gamma = model.coef_[0] # 0 for the first coef with respect to volume_signed

#Let's compute predicted value 
y_pred = model.predict(X_gamma)
# Let's compute R2 and Mean Squared Error
mse_gamma = mean_squared_error(y_gamma, y_pred)
r2_gamma = r2_score(y_gamma, y_pred)


#definition of regression parameter for eta
X_eta = np.column_stack((volume_signed, volume_signed_squared)) # we want to look at impact of order two
y_eta = data_delta_price_eta
#Fitting model for eta
model.fit(X_eta,y_eta)
#Extraction of coefficient directeur but here at order of 2
eta = model.coef_[1] * tau # 1 for the second with respect to volume_signed_squared and due to quadratic approximation we have to divide by tau
#Let's compute predicted value 
y_pred = model.predict(X_eta)
# Let's compute R2 and Mean Squared Error
mse_eta = mean_squared_error(y_eta, y_pred)
r2_eta = r2_score(y_eta, y_pred)

# Printing all our computed & defined parameters
print(f"Estimated Gamma: {gamma}")
print(f"Estimated Eta: {eta}")
print("Sigma:", sigma)
print("Epsilon:", epsilon)
print("Tau:", tau)
print("Lambda", lam)
#Printing our computed measure to analyze our model
print(f'MSE_gamma : {mse_gamma}')
print(f'r2_gamma : {r2_gamma}')
print(f'MSE_eta : {mse_eta}')
print(f'r2_eta: {r2_eta}')

# determination of K coefficient using the hypothesis that tau = 1/24 is sufficiently near of zero in order to have this simplification
K = np.sqrt(((sigma**2) * lam) / (eta)) # Here we note that lambda has to be positive

def trajectory(X, T, K, tau, data):
    ans = []
    indexer = 0
    for t in range(T):
        if (data['Date'].values.astype(float))[t] >= indexer * tau: #This condition make sure that we sell a certain amount of our portfolio only every hour
            indexer = indexer + 1
            x = ((np.sinh(K * (T - (t - (1 / 2) * tau))) / np.sinh(K * T) * X))
            ans.append(x)
        else:
            x = ans[-1] if ans else 0  # Use 0 if ans is empty but normaly no because we start at t_0 for the first liquidation
            
    return np.array(ans)

trajectory_X0_K0 = trajectory(X_liquidate, T, K, tau, data_copy) #Here we use data_copy because data has been corrupted due to too many modification on it's number of row

#Definition of the amount to trade every hour up to lambda = 2E-6 and X = 1000
def Strategie(trajectory):
    strat = []
    for i in range(0, len(trajectory) - 1): # we can't consider last index as it's a rest
        diff = trajectory[i] - trajectory[i+1] 
        strat.append(diff)
    strat.append(trajectory[len(trajectory)-1]) # we add the rest
    return strat
strategie = Strategie(trajectory_X0_K0)
strategie_trade = strategie[:len(strategie)-1]
rest = strategie[len(strategie)-1:len(strategie)]
print(f"\n Trade to be made every hour in order {strategie_trade}\n")
print(f"Rest at the end of trade : {rest}")

# Plot the trajectory
plt.plot(trajectory_X0_K0, label='Optimal Liquidation Trajectory', color='red')

plt.xlabel('Time')
plt.ylabel('Number of shares held in dollars')
plt.title('Optimal Liquidation Trajectory and Original Data Points')
plt.legend()

# Set x-axis ticks to be one by one
plt.xticks(range(len(trajectory_X0_K0)))

plt.show()