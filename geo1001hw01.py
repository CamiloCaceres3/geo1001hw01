#-- GEO1001.2020--hw01
#-- Camilo Caceres
#-- [532210]
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import scipy.stats as st
 
#Read the xls and make a pandas dataframe
heatA = pd.read_excel('HEAT - A_final.xls', skiprows=[0,1,2,4])
heatA = heatA.rename(columns = {'Direction ‚ True': 'Direction', 'Direction ‚ Mag' : 'Direction Mag'} )
heatA.name = 'HeatA'
heatB = pd.read_excel('HEAT - B_final.xls', skiprows=[0,1,2,4])
heatB = heatB.rename(columns = {'Direction ‚ True': 'Direction', 'Direction ‚ Mag' : 'Direction Mag'})
heatB.name = 'HeatB'
heatC = pd.read_excel('HEAT - C_final.xls', skiprows=[0,1,2,4])
heatC = heatC.rename(columns = {'Direction ‚ True': 'Direction', 'Direction ‚ Mag' : 'Direction Mag'})
heatC.name = 'HeatC'
heatD = pd.read_excel('HEAT - D_final.xls', skiprows=[0,1,2,4])
heatD = heatD.rename(columns = {'Direction ‚ True': 'Direction', 'Direction ‚ Mag' : 'Direction Mag'})
heatD.name = 'HeatD'
heatE = pd.read_excel('HEAT - E_final.xls', skiprows=[0,1,2,4])
heatE = heatE.rename(columns = {'Direction ‚ True': 'Direction', 'Direction ‚ Mag' : 'Direction Mag'})
heatE.name = 'HeatE'

#Get the measures signs of the columns of the data frames.
headers = pd.read_excel('HEAT - A_final.xls', skiprows=[0,1,2], header =[0,1] )
headers = headers.rename(columns = {'Direction ‚ True': 'Direction', 'Direction ‚ Mag' : 'Direction Mag', 'Unnamed: 1_level_1': chr(176),
                                   '¬∞C': chr(176)+ 'C', '¬∞': chr(176)} )
headers =headers.columns.tolist()
ind = [[f[0]  for f in headers], [f[1]  for f in headers]]
sensorm= pd.DataFrame(ind)
sensorm.columns = heatA.columns
sensorm.drop(0)
#List for the colors of the graphs
sensorcolor = ['darkblue','darkorange', 'darkgreen', 'red', 'darkmagenta' ]
#List of the pandas dataframes of the sensors
sensorlist = [heatA, heatB, heatC, heatD, heatE]
#Initial Parameters of plots
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (17, 6),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

#A1
#A1
#Function to calculate de mean statistics and save it into a dataframe
def meanstatistics(df):
    stats = pd.DataFrame()
    stats['Mean'] = df.mean()
    stats['Variance'] = df.var()
    stats['Standard Deviation'] = df.std()
    return stats

#Write the mean statistis into a excelfile with each sensor in different excel sheet
i=0
with pd.ExcelWriter('meanstatistcs.xlsx') as writer:
    for frame in sensorlist:
        meanstatistics(frame).to_excel(writer, sheet_name= frame.name)
        i+=1

plt.rcParams["figure.figsize"] = [17,10]
#A1
#Function that makes a subplot with histograms given a list of data frames, a variable and numer of bins
def histograms(lists, var, bins):
    fig, axes = plt.subplots((len(lists)-1)//2, 3, constrained_layout=True)
    fig.suptitle('Histograms', fontsize=30)
    axes = axes.flatten()
    fig.delaxes(axes[-(((len(lists)-1)//2 *3)%len(lists))])
    i = 0 
    for frame in lists:
        axes[i].hist(frame[var],bins = bins, alpha = 0.3, rwidth= 0.85, label = frame.name, color = sensorcolor[i] )
        axes[i].legend()
        axes[i].set_title(var +' Histogram with '+str( bins)+' bins')
        axes[i].set_xlabel(var+ ' [' + sensorm[var][1]+']')
        axes[i].set_ylabel('Frequency')
        i +=1
    plt.savefig('../hw01/histogramsT'+ str(bins) +'bins.png')
    plt.show()
#Make histograms 
histograms(sensorlist, 'Temperature' , 50)
histograms(sensorlist, 'Temperature' , 5)

#A1
#Function to make the cumulative frequency polygon given a dataframe and a variable. It returns a dataframe with the freq. polygon
def acfrequency_pol(df, var):
    df[var].sort_values()
    n = df[var].count()
    bins = round(2 * n**(1/3))
    c = df[var].value_counts(sort = False , bins = round(2 * n**(1/3)) )
    h = c.sort_index()
    k = pd.DataFrame({var:h.index, 'Frequency':h.values})
    k['Cumulative Frequency'] = k['Frequency'].cumsum()
    k[ var +' Bin'] = 0.0 
    for i in range(len(k)):
        k.loc[i,var +' Bin'] =k.loc[i,var].mid
    return k

#Function to make the  frequency polygon given a dataframe and a variable. It returns a dataframe with the freq. polygon
def frequency_pol(df, var):
    df[var].sort_values()
    n = df[var].count()
    bins = round(2 * n**(1/3))
    c = df[var].value_counts(sort = False , bins = round(2 * n**(1/3)) )
    h = c.sort_index()
    k = pd.DataFrame({var:h.index, 'Frequency':h.values})
    for i in range(len(h)):
        k.loc[i,var +' Bin'] =k.loc[i,var].mid
    return k
#Function to plot the cumulative frequency polygon 
def plot_accfrequency_polygons(lists,var):
    for frame in lists:
        s = acfrequency_pol(frame,var)
        s.name = frame.name
        r =plt.plot( s['Cumulative Frequency'], 'o-', label = str(frame.name))
        plt.title('Cumulative Frequency Polygons of ' +var,  fontsize=30)
        plt.legend()  
        plt.xlabel('Temperature' + ' [' + sensorm[var][1]+']')
        plt.ylabel('Cumulative Frequency')
    plt.savefig('../hw01/Cumulative_Frequency_Polygons.png')
    plt.show()
#Function to plot the frequency polygon 
def plot_frequency_polygons(lists,var):
    for frame in lists:
        s = frequency_pol(frame,var)
        s.name = frame.name
        r =plt.plot( s['Frequency'], 'o-', label = str(frame.name))
        plt.title('Frequency Polygons of ' +var,  fontsize=30)
        plt.legend()  
        plt.xlabel('Temperature' + ' [' + sensorm[var][1]+']')
        plt.ylabel('Frequency')
    plt.savefig('../hw01/Frequency_Polygons.png')
    plt.show()

# PLot the cumulative frequency polygon and frequeny polygon
plot_accfrequency_polygons(sensorlist,'Temperature')
plot_frequency_polygons(sensorlist,'Temperature')
  
plt.rcParams["figure.figsize"] = [17,20]
#Function to make a boxplot given a variable and a list of sensor dataframes
def box_plot(nam, lists):
    fig, axes = plt.subplots(len(nam),1, constrained_layout=True)
    fig.suptitle('Box Plots', fontsize=30)
    axes = axes.flatten()
    j=0
    for vari in nam:
        i=0
        df = pd.DataFrame()
        for frame in lists:
            df.insert(i, frame.name, frame[vari])
        ax = sns.boxplot(data=df, ax = axes[j] )
        ax.set(ylabel = vari +' [' + sensorm[vari][1]+']')
        ax.set_title('Box Plot of ' +vari  , fontsize = 30)
        ax.grid(axis = 'y')           
        j+=1
    plt.savefig('../hw01/Boxplots.png')
    plt.show()
#Boxplot
sensorlistA = [heatE, heatD,heatC, heatB, heatA]
var = ['Temperature', 'Wind Speed','Direction']
box_plot(var, sensorlistA)

#A2
plt.rcParams["figure.figsize"] = [17,20]
#Function to plot diferent pmf given a list of sensor dataframes and a variable
def pmf(lists, var):
    fig, axes = plt.subplots(3,2, constrained_layout=True)
    fig.suptitle(var +' Probability Mass Functions', fontsize=30)
    axes = axes.flatten()
    fig.delaxes(axes[-(((len(lists)-1)//2 *3)%len(lists))])
    i = 0
    for frame in lists:
        c= frame[var].value_counts()
        p = c/len(frame[var])
        d = p.sort_index()
        axes[i].bar(d.index,d, width = 0.1 ,linewidth = 1, edgecolor  = 'black', color = sensorcolor[i] )
        axes[i].set_xlabel(var + ' [' + sensorm[var][1]+']')
        axes[i].set_ylabel('Probability')
        axes[i].set_title('PMF of ' + frame.name)
        i +=1
    plt.savefig('../hw01/PMF.png')
    plt.show()

#Plot the pm
pmf(sensorlist, 'Temperature')


plt.rcParams["figure.figsize"] = [17,20]
#Functions that plots the pdf given a lists of sensor dataframes and a variable
def pdf1(lists, var):
    fig, axes = plt.subplots(3,2, constrained_layout=True)
    fig.suptitle(var +' Probability Density Functions', fontsize=30)
    axes = axes.flatten()
    fig.delaxes(axes[-(((len(lists)-1)//2 *3)%len(lists))])
    i = 0
    for frame in lists:
        a1 =axes[i].hist(x =frame[var],bins=25,density=True, color=sensorcolor[i],alpha=0.7, width =0.85, label = 'PDF')
        c = [a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2,a1[0]]
        axes[i].plot(c[0], c[1], color='k')
        axes[i].set_xlabel(var + ' [' + sensorm[var][1]+']')
        axes[i].set_ylabel('Probability')
        axes[i].set_title('PDF of ' + frame.name)
        i +=1
    plt.savefig('../hw01/PDF.png')
    plt.show()
    
#Plot pdf
pdf1(sensorlist, 'Temperature')

#Function to plot the cdf given a list of sensor data frame and a variable
def cdf(lists, var):
    fig, axes = plt.subplots(3,2, constrained_layout=True)
    fig.suptitle(var + ' Cumulative Density Functions', fontsize=30)
    axes = axes.flatten()
    i = 0
    for frame in lists:
        a1= axes[i].hist(x =frame[var],bins=25,density=True, cumulative = True, color=sensorcolor[i],alpha=0.7, width =0.7, label = 'PDF')
        c = [a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2,a1[0]]
        axes[i].plot(c[0], c[1], color='k')
        axes[-1].plot(c[0], c[1], color=sensorcolor[i], label = frame.name)
        axes[-1].legend()
        axes[i].set_xlabel(var + ' [' + sensorm[var][1]+']')
        axes[i].set_ylabel('Probability')
        axes[i].set_title('CDF of ' + frame.name)
        i +=1
    axes[-1].set_xlabel(var + ' [' + sensorm[var][1]+']')   
    axes[-1].set_ylabel('Probability')
    axes[-1].set_title('CDF of al Sensors')
    plt.savefig('../hw01/CDF.png')
    plt.show()
#Plot CDF
cdf(sensorlist, 'Temperature')

plt.rcParams["figure.figsize"] = [17,20]
#Function that plots the pdf with de kde distribution given a lists of sensors dataframes and a variable
def pdf_kernel(lists, var):
    fig, axes = plt.subplots(3,2, constrained_layout=True)
    fig.suptitle('PDF and Kernel Density Estimation', fontsize=30)
    axes = axes.flatten()
    fig.delaxes(axes[-(((len(lists)-1)//2 *3)%len(lists))])
    i = 0
    for frame in lists:
        n, x , _  = axes[i].hist(x=frame[var],bins=25, density = True,
                    color=sensorcolor[i],alpha=0.7, label = 'PDF', edgecolor='k')
        sns.kdeplot(frame[var], 
                   ax = axes[i], label = 'Kernel Estimation Seaborn', color = 'k')
        density = st.gaussian_kde(frame[var])
        axes[i].plot(x,density(x), color = 'yellow', label = 'Kernel Estimation Scipy')
        #frame[var].plot.kde(ax = axes[i])
        #frame[var].plot.density( ax = axes[i])
        #axes[i].set_xlim(xmin=-10)
        axes[i].set_xlabel(var + ' [' + sensorm[var][1]+']')
        axes[i].legend()
        axes[i].set_ylabel('Probality')
        axes[i].set_title('PDF of ' + frame.name)
        i +=1
    plt.savefig('../hw01/wind_speed_kpdf.png')
    plt.show()

pdf_kernel(sensorlist, 'Wind Speed')

#A3

#Function that makes a dataframe with one variable and with all the sensors.
def df_variable(lists,var):
    df = pd.DataFrame()
    i=0
    for frame in lists:
        df.insert(i, frame.name, frame[var])
        i +=1
    return df
#Function that calculates the pearson and spearman correlations, writes it in an excel file and return the two dataframes
#given a list of dataframes and a variable
def corr(lists, var):
    df = df_variable(sensorlist, var)
    pearson = df.corr(method='pearson')
    pearson.name = 'Pearson'
    spearman = df.corr(method = 'spearman')
    spearman.name = 'Spearman'
    with pd.ExcelWriter('corr'+var+'.xlsx') as writer:
        pearson.to_excel(writer, sheet_name= var+' Pearson')
        spearman.to_excel(writer, sheet_name= var+' Spearman')
    return [pearson, spearman]
#Calculates the correlation of the sensor lists daa frame and a specific variable
corr(sensorlist,'Temperature')        
corr(sensorlist, 'WBGT')
corr(sensorlist,'Crosswind Speed')

#Plots a scatter plot between a list data frame and a given variable
def corrplot(lists, var):
    plt.rcParams["figure.figsize"] = [20,7]
    z=0
    for frame in lists: 
        a = pd.DataFrame( columns= ['x','y'])
        y = 0
        for i in frame.index:
            x = 0
            for c in frame.columns:
                if x >y:
                    a = a.append({'x': i+c, 'y': frame.iloc[x][y]}, ignore_index=True)
                x+=1
            y +=1
            if z==0: 
                c = 'Pearson'
            else:
                c = 'Spearman'
        z+=1
        plt.scatter(x = a.x , y=a.y, label = c)
        plt.xlabel('Sensor Pairs')
        plt.ylabel('Correlation')
        plt.title('Correlation Scatter Plot for ' +var)
        plt.legend()
        plt.savefig('../hw01/scatter'+var+'.png')
    plt.show()
#Plots the correlation scatter plot
corrplot(corr(sensorlist,'Temperature'),'Temperature')
corrplot(corr(sensorlist,'WBGT'), 'WBGT')
corrplot(corr(sensorlist,'Crosswind Speed'), 'Crosswind Speed')

plt.rcParams["figure.figsize"] = [17,8]

#A4

#PLots different cdf in a subplot for a list of dataframs given and a list of variables given
def cdf_all(lists ,col):
    j=0
    fig, axes = plt.subplots(1,2, constrained_layout=True)
    fig.suptitle(' Cumulative Density Functions', fontsize=30)
    axes = axes.flatten()
    fig2, axes2 = plt.subplots(1,2, constrained_layout=True)
    fig2.suptitle('Histograms', fontsize=30)
    axes2 = axes2.flatten()
    for columns in col: 
        var = columns
        if columns!= 'FORMATTED DATE-TIME':
            i = 0
            for frame in lists:
                a1= axes2[j].hist(x =frame[var],bins=25,density=True,cumulative = True, color=sensorcolor[i],alpha=0.2, label = 'PDF')
                c = [a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2,a1[0]]
                axes[j].plot(c[0], c[1], color=sensorcolor[i], label = frame.name, linewidth = 5)
                axes[j].legend()
                axes[j].set_xlabel(var+ ' [' + sensorm[var][1]+']')
                axes[j].set_ylabel('Probability')
                axes[j].set_title('CDF of ' + var + ' [' + sensorm[var][1]+']')
                i +=1    
            j +=1
    fig.savefig('../hw01/cdf4.png')
#Plots cdf
col = ['Temperature', 'Wind Speed']
cdf_all(sensorlist, col)

#Function that calculates a confidence interval given a list of variables and a list of data frames return a dataframe with the answer
def ic_all(lists, col):
    df = pd.DataFrame()
    for columns in col:
        if columns!= 'FORMATTED DATE-TIME':
            i=1
            df2 = pd.DataFrame()
            df2.insert(0, 'Variable',[columns])
            for frame in lists:
                ic = st.t.interval(0.95, len(frame[columns])-1, loc=np.mean(frame[columns]), scale=st.sem(frame[columns]))
                df2.insert(i, frame.name + ' Confidence Interval' , str( ic))
                i +=1
            df = df.append(df2, ignore_index=True)
            df2.drop(0)
    df =df.set_index('Variable')       
    df =df.transpose()            
    return df

#Calculates the ic
col = ['Temperature', 'Wind Speed']
intconf  = ic_all(sensorlist, col)
intconf.to_csv("confidence_intervals.csv")
intconf

resdf = pd.DataFrame(columns= ['Sensor1','Sensor2', 'Variable', 't value', 'p value'])
#function that given dataframe 1 and dataframe 2, with the variable var1 calculates a t test and returns a df given with the answer on it
def test(list1,list2, var1, df):
    t , p = st.ttest_ind(list1[var1], list2[var1])
    df = pd.DataFrame(columns= ['Sensor1','Sensor2', 'Variable', 't value', 'p value'])
    df = df.append({'Sensor1': list1.name, 'Sensor2': list2.name , 'Variable': var1, 
                  't value': t , 'p value': p}, ignore_index=True) 
    return df

#Calculates de test of given sensors and variables
edt = test(heatE, heatD, 'Temperature',resdf)
resdf = resdf.append(edt)
edw = test(heatE, heatD, 'Wind Speed',resdf)
resdf = resdf.append(edw)
dct = test(heatD, heatC, 'Temperature',resdf)
resdf = resdf.append(dct)
dcw = test(heatD, heatC, 'Wind Speed',resdf)
resdf = resdf.append(dcw)
cbt = test(heatC, heatB, 'Temperature',resdf)
resdf = resdf.append(cbt)
cbw = test(heatC, heatB, 'Wind Speed',resdf)
resdf = resdf.append(cbw)
bat = test(heatB, heatA, 'Temperature',resdf)
resdf = resdf.append(bat)
baw = test(heatB, heatA, 'Wind Speed',resdf)
resdf = resdf.append(baw)
resdf.to_csv("ttest.csv")
resdf

#Bonus Question

#Concatenate in one Dataframe all sensors since they don't have any differences.
sensorlist = [heatA,heatB, heatC, heatD, heatE]
df = pd.concat(sensorlist)
# Formatting Date to timestamp
pd.to_datetime(df['FORMATTED DATE-TIME'], infer_datetime_format=True)
#Calculate the  average temperature between sensors truncating the timestamp to minute.
byhour = df.groupby([df['FORMATTED DATE-TIME'].dt.floor('min')])['Temperature'].mean()
byhourinone = pd.DataFrame(byhour)
byhourinone = byhourinone.reset_index()
#Count the number of data points to calculate the average daily temperature (has to be 120 for every date)
bydaycount = byhourinone.groupby([byhourinone['FORMATTED DATE-TIME'].dt.date])['Temperature'].count()
bydaycount = pd.DataFrame(bydaycount)
bydaycount = bydaycount.reset_index()
#Calculate de daily average temperature
bydaymean = byhourinone.groupby([byhourinone['FORMATTED DATE-TIME'].dt.date])['Temperature'].mean()
bydaymin = byhourinone.groupby([byhourinone['FORMATTED DATE-TIME'].dt.date])['Temperature'].min()
bydaymax = byhourinone.groupby([byhourinone['FORMATTED DATE-TIME'].dt.date])['Temperature'].max()
bydaymean = pd.DataFrame(bydaymean)
bydaymean = bydaymean.reset_index()
bydaymin = pd.DataFrame(bydaymin)
bydaymin = bydaymin.reset_index()
bydaymax = pd.DataFrame(bydaymax)
bydaymax = bydaymax.reset_index()
# Join the table count to the mean
new = bydaymean.join(bydaycount,  rsuffix='_count')
new = new.join(bydaymax, rsuffix= '_max')
new = new.join(bydaymin, rsuffix= '_min')
new['Mean Temperature'] = new.apply(lambda row: (row['Temperature_max'] + row['Temperature_min'])/2, axis=1)
# Filter data frame in order to have only the days with 72 data points
new = new[new['Temperature_count'] == 72]
#Calculate de day of min Average daily temperature 
min1 =new['Temperature'].idxmin()
#Calculate de day of max average daily temperature 
max1 =new['Temperature'].idxmax()
print( 'The coolest day is: ' + str( new.iloc[min1]['FORMATTED DATE-TIME']) + ' with an Average temperature of: ' + str( new.iloc[min1]['Temperature']))
print('The warmest day is: '+ str(new.iloc[max1]['FORMATTED DATE-TIME']) + ' with an Average temperature of: ' + str( new.iloc[max1]['Temperature']))

#Calculate de day of min Mean daily temperature 
min1 =new['Mean Temperature'].idxmin()
#Calculate de day of max Mean daily temperature 
max1 =new['Mean Temperature'].idxmax()
print('The coolest day is: ' + str( new.iloc[min1]['FORMATTED DATE-TIME']) + ' with a Mean temperature of: ' + str( new.iloc[min1]['Mean Temperature']))
print('The warmest day is: ' +str(new.iloc[max1]['FORMATTED DATE-TIME']) + ' with a Mean temperature of: ' + str( new.iloc[max1]['Mean Temperature']))

