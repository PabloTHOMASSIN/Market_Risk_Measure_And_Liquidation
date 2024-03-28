import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

#Preprocessiing
data = pd.read_excel(r"C:\Users\pablo\OneDrive - De Vinci\COURS\A4\S1\Market Risk\TD\TD_5\Dataset TD5.xlsx")
data = data.drop(data.index[0]) # we will drop first two row as it's not pure data
data = data.drop(data.index[0])

data.columns = ['GBPEUR_Date', 'GBPEUR_HIGH', 'GBPEUR_LOW', 'Unamed_1', 'SEKEUR_Date',
       'SEKEUR_HIGH', 'SEKEUR_LOW', 'Unamed_2', 'CADEUR_Date', 'CADEUR_HIGH',
       'CADEUR_LOW']
data = data.drop('Unamed_1', axis=1) # Due to spaces between data there is empty column to drop
data = data.drop('Unamed_2', axis=1)

#Let's print our data
print(data)


#Let's compute average price and create new column in our data set to use them here we use numpy built in function to preserve as much as possible significative numbers
data['GBPEUR_Avg_Price'] = np.divide(np.add(data['GBPEUR_HIGH'] ,data['GBPEUR_LOW']),2)
data['SEKEUR_Avg_Price'] = np.divide(np.add(data['SEKEUR_HIGH'] ,data['SEKEUR_LOW']), 2)
data['CADEUR_Avg_Price'] = np.divide(np.add(data['CADEUR_HIGH'] ,data['CADEUR_LOW']),2)
#Let's observe new data set
print(data)

#Let's define a function to compute returns and creating new column for each of our Fx once again we use numpy function to preserve significative number
def Returns(data, column, stocks):

    returns_name = 'Returns_' + str(stocks)
    data[returns_name] = np.divide(np.subtract(data[column] ,data[column].shift(1)),data[column].shift(1))
    print(data)
    return data
#Let's apply it
data = Returns(data, 'CADEUR_Avg_Price', 'CADEUR')
data = Returns(data, 'GBPEUR_Avg_Price', 'GBPEUR')
data = Returns(data, 'SEKEUR_Avg_Price', 'SEKEUR')
#Let's observe our data
print(data)
# Due to Nan value on first row let's avoid it by dropping it
data = data.dropna()
print(data)
# Now let's define our mother walette with respect to the cours page 302
def haar_mother_wavelet(x):
    return np.where((x >= 0) & (x < 0.5), 1, np.where((x >= 0.5) & (x < 1), -1, 0))

#Here is the transform function obtained from the course 
def Haar_transform(data, t):
    haar_transform = []
    N = len(data) # The total number of data available

    for k in range(N // t): #We use floor division which is a division that takes the rounded number after dividing
        start_idx = k * t # begining of value considered
        end_idx = start_idx + t # end of value considered with respect to t the time scale
        scale_factor = 1 / np.sqrt(t)

        # Let's compute approximation coefficient
        approx_coeff = scale_factor * np.sum(haar_mother_wavelet(np.arange(start_idx, end_idx) / N) * data[start_idx:end_idx]) # np.arrange is there to make sure we are taking our value in the good order
        haar_transform.append(approx_coeff)

        # Let's compute detail coefficient
        detail_coeff = scale_factor * np.sum(haar_mother_wavelet(np.arange(start_idx + 0.5 * t, end_idx + 0.5 * t) / N) * data[start_idx:end_idx])
        haar_transform.append(detail_coeff)

    # Approximation coefficients for the remaining parts that is if N is a multiple of our time scales only
    if N % t != 0:
        remaining_coeff = scale_factor * np.sum(haar_mother_wavelet(np.arange(N - (N % t), N) / N) * data[-(N % t):]) / (N % t)
        haar_transform.extend([remaining_coeff] * (N % t))# Here we add elements at the end of the list

    return np.array(haar_transform)# we give the result under the form of a numpy.array which will be usefull later

t=int(30/15) # We take a time intervalle of 30 minuts and due to our data spacing being 15 minuts we divide by this here for the begining t = 2 which will give a really precise result of our modelisation

# Apply Haar wavelet transform to returns we make sure to pass numpy array with .values and with .astype(float) we make sure the type of the data is float
haar_transform_CADEUR_mother = Haar_transform(data['Returns_CADEUR'].values.astype(float), t)
haar_transform_GBPEUR_mother = Haar_transform(data['Returns_GBPEUR'].values.astype(float), t)
haar_transform_SEKEUR_mother = Haar_transform(data['Returns_SEKEUR'].values.astype(float), t)

# Create a 3x2 grid of subplots to print our result
fig, axs = plt.subplots(3, 2, figsize=(14, 10))

# Plot for GBPEUR
axs[0, 0].plot(data['GBPEUR_Date'], data['Returns_GBPEUR'])
axs[0, 0].set_title('Returns - GBPEUR')

# Convert datetime to numerical values due to the possibility of t = 1 we have to make sure the date value correspond to the good haar value obtained
date_values_GBPEUR = mdates.date2num(data['GBPEUR_Date'])
haar_x_values_GBPEUR = np.linspace(date_values_GBPEUR[0], date_values_GBPEUR[-1], len(haar_transform_GBPEUR_mother))
axs[0, 1].step(haar_x_values_GBPEUR, haar_transform_GBPEUR_mother, color='red', label='Haar Mother')
axs[0, 1].set_title('Haar Transform mother - GBPEUR')
axs[0, 1].legend()
# Plot for CADEUR
axs[1, 0].plot(data['CADEUR_Date'], data['Returns_CADEUR'])
axs[1, 0].set_title('Returns - CADEUR')
# Convert datetime to numerical values
date_values_CADEUR = mdates.date2num(data['CADEUR_Date'])
haar_x_values_CADEUR = np.linspace(date_values_CADEUR[0], date_values_CADEUR[-1], len(haar_transform_CADEUR_mother))
axs[1, 1].step(haar_x_values_CADEUR, haar_transform_CADEUR_mother, color='red', label='Haar Mother')
axs[1, 1].set_title('Haar Transform mother - CADEUR')
axs[1, 1].legend()
# Plot for SEKEUR
axs[2, 0].plot(data['SEKEUR_Date'], data['Returns_SEKEUR'])
axs[2, 0].set_title('Returns - SEKEUR')
# Convert datetime to numerical values
date_values_SEKEUR = mdates.date2num(data['SEKEUR_Date'])
haar_x_values_SEKEUR = np.linspace(date_values_SEKEUR[0], date_values_SEKEUR[-1], len(haar_transform_SEKEUR_mother))
axs[2, 1].step(haar_x_values_SEKEUR, haar_transform_SEKEUR_mother, color='red', label='Haar Mother')
axs[2, 1].set_title('Haar Transform mother - SEKEUR')
axs[2, 1].legend()
# Add a common y-axis label
for ax in axs.flat:
    ax.set(ylabel='Value')
# Adjust layout to prevent clipping of ylabel
fig.tight_layout()
# Show the plot
plt.show()



#Now let's introduce differents time scales with respect to 10080 minuts representing a week. But our data set starting not at 00:00:00 we will consider a little bit more of value than necessary
time_minutes = [15, 30, 60, 120, 240, 480, 960, 1440, 2880, 4320, 10080]
scales = [int(i/15) for i in time_minutes]
print(f"Our different time scales for the haar wavelette {scales}")


# Calculate portfolio prices by taking the sum of the three avg FX pric and dividing it by 3
data['Prices_Portfolio'] = (data['GBPEUR_Avg_Price'] + data['SEKEUR_Avg_Price'] + data['CADEUR_Avg_Price']) / 3
data = Returns(data, 'Prices_Portfolio', 'Portfolio') # Then we compute their returns (average on the normalized of the average price of the three FX)
data = data.dropna()
print(data)


#Let's now define correlation matrix for each scale defined earlier
def wavelet_correlation_matrix(data, scales):
    correlation_matrices = []

    for scale in scales:
        # Apply Haar wavelet transform to portfolio returns
        haar_transform_portfolio = Haar_transform(data, scale)
        # Combine approximation and detail coefficients for correlation calculation because we want to look at the correlation of this two coefficient to understand up and down of our data
        half_len = len(haar_transform_portfolio) // 2 #Depending of the half length of the data set there is disjunction cas in order to avoir division by zero
        if len(haar_transform_portfolio) % 2 == 0:
            combined_coefficients = np.vstack([
                haar_transform_portfolio[:half_len],
                haar_transform_portfolio[half_len:]
            ])
        else:
            combined_coefficients = np.vstack([
                haar_transform_portfolio[:half_len],
                haar_transform_portfolio[half_len:-1]
            ])
        #We used Vstack to create a vertical array of both our coefficient
        # Calculate the correlation matrix for the current scale
        correlation_matrix = np.corrcoef(combined_coefficients) #Here we use numpy.corrcoef, which is using a covariance classical formula and dividing it by the var of both the two set considered here our splitted set

        # We had each matrix of different scales in a bigger vector to have them at our disposition in order
        print(correlation_matrix)
        correlation_matrices.append(correlation_matrix)
    #We return all the matrices in order
    return correlation_matrices

#Let's compute all the correlation matrices with once again a numpy array given on the returns of portfolio
Correlation_Matrices = wavelet_correlation_matrix(data['Returns_Portfolio'].values.astype(float), scales)
print(f"Here is the correlation matrix: {Correlation_Matrices}")

#With respect to the formula page 263 of the cours
def Hurst_Exponent(data):
    N = 1 #Spacinng variable to make [0, T] sliced with respect to 1/N
    k = 2 # With the respect to fomula page 263 involving moment of order 2
    T = len(data)
    M_2, M_2_prime = 0, 0
    # We compute the two empirical absolute moments and make sure to round up indexs in order to make them coeherent with a discrete selection
    for i in range(0, int(N*T)):
        M_2 += abs(data[int(i/N)] - data[int((i-1)/N)])**k
    for i in range(0, int(N*T/2)):
        M_2_prime += abs(data[int(2*i/N)] - data[int(2*(i-1)/N)])**k
        
    M_2 = M_2 / (N*T)
    M_2_prime = (2 / (N*T) ) * M_2_prime
    
    return 0.5 * np.log2(M_2_prime/M_2) 


# Calculate Hurst exponent with Prices normalized
hurst_value = Hurst_Exponent(data['Prices_Portfolio'].values.astype(float))
print(f"Hurst Exponent: {hurst_value}")

# Function to calculate volatility vector with scaling based on Hurst exponent
def Volatility_vector(data, hurst, scales):
    sigma = data.std() # Here we use the standard deviation included in numpy looking at the sqrt of the var
    volatility_vector = [sigma * (scale**hurst) for scale in scales] #Each member of the vector is the given volatility for the given scale (which is the same that is used for our haar function) up to a common hurst exponent 
    return volatility_vector



# Calculate volatility vector on Returns of the portfolio
volatility_vector = Volatility_vector(data['Returns_Portfolio'].values.astype(float), hurst_value, scales)
print(f"Volatility Vector: {volatility_vector}")


#Let's define our covariance matrix with respect to mathematical formula
def Covariance_Matrices(Correlation_Matrices, volatility_vector, scales):
    num_matrices = len(Correlation_Matrices) # the number of correlation matrix we need to itterate uppon which is the same number as the len(scales)
    covariance_matrices = [] #stocking vectors for each covariance matrix in order with respect to scales

    for i in range(num_matrices):
        vol_matrix = np.diag([volatility_vector[i], volatility_vector[i]]) # If we recall that we have symmetrical correlation matrices we need to developp a 2X2 diagonal matrix made of on the diagonal of the probability for the given scales, which is the case because they are all ordered in the same way.
       
        correlation_matrix = Correlation_Matrices[i] # We select a Correlation Matrix
        
        # Matrix multiplication to get the covariance matrix with np.dot which is used for multiplying matrix (for information it's using hadamard product) more over because vol_matrix is diagonal and of the same six as correlation_matrix we don't need to take the transpose at the end
        covariance_matrix = np.dot(vol_matrix, np.dot(correlation_matrix, vol_matrix))
        #We stock each covariance matrix in a vector
        covariance_matrices.append(covariance_matrix)

        # Display the covariance matrix using seaborn, this module offer the opportunity to easly print heat map which are more readable than any other things
        sns.heatmap(covariance_matrix, annot=True, cmap="coolwarm", linewidths=.5,
                   xticklabels=['Approximation', 'Details'], yticklabels=['Approximation', 'Details'])
        plt.title(f"Covariance Matrix - Scale =  {scales[i]}") # here it will print the scale considered so t = int(15/15) = 1 for example
        plt.show()
    #Let's return all our covariance_matrices
    return covariance_matrices
#Let's compute the covariance matrices for our already computed parameter 
covariance_matrices = Covariance_Matrices(Correlation_Matrices, volatility_vector, scales)

# Access individual covariance matrices and print them to make sure the result is coherent with seaborn heatmap print
for i, cov_matrix in enumerate(covariance_matrices):
    print(f"Covariance Matrix - Scale = {scales[i]}:\n{cov_matrix}")
    

for scale in scales:
    hurst_value = Hurst_Exponent(data['Prices_Portfolio'].values.astype(float)[:scale])
    print(f"Hurst Exponent for Time Scale {scale}: {hurst_value}")
    
    

def Overlapping_Returns(arr, points_shared):
    returns = []
    n = len(arr)

    for i in range(n - points_shared + 1):# we consider this sum until this index in order for points to make groups of points_shared
        subset1 = arr[i:i + points_shared]# we define the price at t for a certain numbers of points_shared
        subset2 = arr[i + 1:i + points_shared + 1]# we define the price at t +1  for a certain numbers of points shared)
        
        # Calculate returns for the overlapping subsets
        returns.append((subset2[-1] - subset1[0]) / subset1[0])

    return np.array(returns)

def calculate_volatility(data, points_shared, hurst):
    returns = Overlapping_Returns(data, points_shared)  # Compute returns for overlapping points
    volatility = np.std(returns) # standard deviation for the returns series
    
    # Scale volatility based on the Hurst exponent
    scaled_volatility = volatility * (points_shared ** hurst) # here we choosed to used point_shared as the scaling factor in order to catch a better response on short term analysis but if we wanted to look at longer term we could have considered scaling = 64

    return scaled_volatility

scaling = 64 #the length of each segment
shared_points = 5 # the number of points shared in price calculation
Price_Portfolio_length = len(data['Prices_Portfolio'].values.astype(float)) # The total length of the column considered
volatility_vector_2 = [] #Stocking array for our volatility

for i in range(0, Price_Portfolio_length, scaling): # here the argument in python go as follow, begin, start, steps
    prices_slice = data['Prices_Portfolio'].values.astype(float)[i:i+scaling] # the sliced interval
    
    #Compute the hurst exponent value for each slice
    hurst_value = Hurst_Exponent(prices_slice)
    
    # Calculate volatility for different time scales with overlapping returns and scaling based on Hurst exponent
    volatility_scaled = calculate_volatility(prices_slice, shared_points, hurst_value)
    
    volatility_vector_2.append(volatility_scaled)

print(f"Here is my volatility for {scaling} points steps considered with returns sharing {shared_points} points for each intervals : {volatility_vector_2}")


# Calculating the weighted average of volatilities here the weight is the same for all the segment because they have the same lenght but we can imagine it's not the case
weighted_volatility_sum = 0
total_points = 0

for i in range(len(volatility_vector_2)):# the size of each segment is scaling
    segment_size = scaling
    total_points += segment_size
    weighted_volatility_sum += segment_size * volatility_vector_2[i]

# Calculating the total volatility as the weighted average
total_volatility = weighted_volatility_sum / total_points

print(f"Total volatility for the entire dataset: {total_volatility}")