# geo1001hw01

This taks consists in 5 points with an inital code to read input files

note: in most of the cases a function is made for plotting in the case that further data is added in the study or other variables are wanted to be plot. (just have to change the arguments!)

# Read Input Files

Having the pandas library we import the  .xls input files to data frames. 
We make another import to have the values of measurements (i.e. for wind speed, m/s). 
We make name the data frames and create a list for colors and put the initial parameters for plots.


# A1

We make a function to save in one dataframe the mean statistics giving a dataframe. Then save it in an excel file.

For histograms a  function is made  to create a subplot with histograms given a list of data frames, a variable and numer of bins.

For Frequency polygons  a Function is made  given a dataframe and a variable. It returns a dataframe with the freq. polygon. Then a function to plot the freq. polygons is made.

For the boxplots a Function is made  given a variable and a list of sensor dataframes.


# A2

A function to create a probability mass function, a probability density function and cumulative density function is made given a sensor data frame and a variable.
The plots are made with subplots.

A function that plots a pdf function and creates a KDE plot is made also with a dataframe and a variable as arguments.

# A3

In the first part of this point we make a function that takes all sensors dataframes given a variable and concatenates it in one table. The result is for example, having all temperature measurements of every sensor y one dataframe.

After that we make a function that takes a list of dataframes and a variable and calculates both pearson and spearman correlation with pandas library function corr. This function in pandas omits the NULL variables in the calculation of the correlations. Therefore we do not have to drop this NULL values.

For scatter plots a function is made in order to take one single combination of the correlations between sensors a plotter with its correlation.

# A4

To calculate a CDF for different sensors in one plot and for different variables, we make a function that takes and list of sensors dataframe and a list of variables to be plot.

Note: This functions helped as the question was changed from 19 variables to 2. For me I had only to change the list of variables that the function reads.

For the confidence intervals, a function is made to print the IC in a cvs file and it calculates the IC given  a list of sensro dataframes and a variable.

For the ttest a function is made, and it print the t and p value from the test given 2 sensor dataframes  which are wanted to be compared and the also the variable. It takes as an argument a empty dataframe in case various pair test are wanted to be saved. 

# Bonus question

For bonus question all the calculations are made froma pandas dataframe point of view. First we concatenate all dataframes in one. Then we make some calulations. (mean, max, min and count) in order to be concise in the results. Then with pandas dataframe functions we calculate the daily temperature and after that se search for the min and max temperature date.






