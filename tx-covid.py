# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 13:38:59 2022

@author: Jason
"""

import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns

from matplotlib.ticker import FuncFormatter

def trends(data,pct=False):
    # we just want the percentage point change for the positivity change
    if pct:
        trendweek = data[-1] - data[-8]
        trend2week = data[-1] - data[-15]
        trendmonth = data[-1] - data[-31]
    else:
        trendweek = (data[-1] - data[-8]) / data[-8]
        trend2week = (data[-1] - data[-15]) / data[-15]
        trendmonth = (data[-1] - data[-31]) / data[-31]
    return(data[-1],data[-8],data[-15],data[-31],trendweek,trend2week,trendmonth)

def makeCSV(filename,data):
    with open(filename,'w',encoding='UTF8',newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
        f.close()

# user settings
debug_mode = False
plot_cases = True
plot_deaths = True
plot_positivity = True
plot_hospital = True

# import case data

cases_df = pd.read_excel('https://dshs.texas.gov/coronavirus/TexasCOVID-19NewCasesOverTimebyCounty.xlsx',\
                         skiprows=[0,1,257,258,259,260],index_col='County')
cases_df.columns = cases_df.columns.str.replace('New Cases ','')
cases_df.columns = pd.to_datetime(cases_df.columns,format='%m-%d-%Y')

# import death data and compute daily deaths (state file gives cumulative)
deaths_df = pd.read_excel('https://dshs.texas.gov/coronavirus/TexasCOVID19DailyCountyFatalityCountData.xlsx',\
                          skiprows=[0,1,257,258],index_col='County Name')
deaths_df.columns = deaths_df.columns.str.replace('Fatalities ','')
deaths_df.columns = pd.to_datetime(deaths_df.columns,format='%m-%d-%Y')
daily_deaths_df = deaths_df.diff(periods=1,axis=1)

# import testing data and compute daily tests (state file gives cumulative)
testing_df = pd.read_excel('https://dshs.texas.gov/coronavirus/TexasCOVID-19CumulativeTestsbyCounty.xlsx',\
                           skiprows=[0,256,257,258],index_col='County')
testing_df.columns = pd.to_datetime(testing_df.columns,format='%Y-%m-%d')
daily_testing_df = testing_df.diff(periods=1,axis=1)

# import hospitalization data
hospital_df = pd.read_excel('https://dshs.texas.gov/coronavirus/CombinedHospitalDataoverTimebyTSA.xlsx',\
    sheet_name='COVID % Capacity',skiprows=[0,1,25,26,27,28],\
    index_col='TSA ID')
hospital_df.index = hospital_df.index.str.replace('.','')
hospital_df.drop(columns=['TSA AREA'],inplace=True)
# there are a few boo-boos in the spreadsheet
hospital_df.columns = hospital_df.columns.str.replace('44051','2020-08-08')
hospital_df.columns = hospital_df.columns.str.replace('.y','')
hospital_df.columns = pd.to_datetime(hospital_df.columns,format='%Y-%m-%d')
hospital_df = hospital_df.replace('%','',regex=True).astype('float')/100

# trauma service areas (TSAs) and names

regions = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',\
           'Q','R','S','T','U','V']

tsa_counties = {
    'A':['Armstrong','Briscoe','Carson','Childress','Collingsworth','Dallam',\
        'Deaf Smith','Donley','Gray','Hall','Hansford','Hartley','Hemphill',\
        'Hutchinson','Lipscomb','Moore','Ochiltree','Oldham','Parmer',\
        'Potter','Randall','Roberts','Sherman','Swisher','Wheeler'],
    'B':['Bailey','Borden','Castro','Cochran','Cottle','Crosby','Dawson',\
        'Dickens','Floyd','Gaines','Garza','Hale','Hockley','Kent','King',\
        'Lamb','Lubbock','Lynn','Motley','Scurry','Terry','Yoakum'],
    'C':['Archer','Baylor','Clay','Foard','Hardeman','Jack','Montague',\
         'Wichita','Wilbarger','Young'],
    'D':['Brown','Callahan','Coleman','Comanche','Eastland','Fisher',\
         'Haskell','Jones','Knox','Mitchell','Nolan','Shackelford',\
         'Stephens','Stonewall','Taylor','Throckmorton'],
    'E':['Collin','Cooke','Dallas','Denton','Ellis','Erath','Fannin',\
         'Grayson','Hood','Hunt','Johnson','Kaufman','Navarro','Palo Pinto',\
         'Parker','Rockwall','Somervell','Tarrant','Wise'],
    'F':['Bowie','Cass','Delta','Hopkins','Lamar','Morris','Red River',\
         'Titus'],
    'G':['Anderson','Camp','Cherokee','Franklin','Freestone','Gregg',\
         'Harrison','Henderson','Houston','Marion','Panola','Rains','Rusk',\
         'Shelby','Smith','Trinity','Upshur','Van Zandt','Wood'],
    'H':['Angelina','Nacogdoches','Polk','Sabine','San Augustine',\
         'San Jacinto','Tyler'],
    'I':['Culberson','El Paso','Hudspeth'],
    'J':['Andrews','Brewster','Crane','Ector','Glasscock','Howard',\
         'Jeff Davis','Loving','Martin','Midland','Pecos','Presidio',\
         'Reeves','Terrell','Upton','Ward','Winkler'],
    'K':['Coke','Concho','Crockett','Irion','Kimble','Mason','McCulloch',\
         'Menard','Reagan','Runnels','Schleicher','Sterling','Sutton',\
         'Tom Green'],
    'L':['Bell','Coryell','Hamilton','Lampasas','Milam','Mills'],
    'M':['Bosque','Falls','Hill','Limestone','McLennan'],
    'N':['Brazos','Burleson','Grimes','Leon','Madison','Robertson',\
         'Washington'],
    'O':['Bastrop','Blanco','Burnet','Caldwell','Fayette','Hays','Lee',\
         'Llano','San Saba','Travis','Williamson'],
    'P':['Atascosa','Bandera','Bexar','Comal','Dimmit','Edwards','Frio',\
         'Gillespie','Gonzales','Guadalupe','Karnes','Kendall','Kerr',\
         'Kinney','La Salle','Maverick','Medina','Real','Uvalde','Val Verde',\
         'Wilson','Zavala'],
    'Q':['Austin','Colorado','Fort Bend','Harris','Matagorda','Montgomery',\
         'Walker','Waller','Wharton'],
    'R':['Brazoria','Chambers','Galveston','Hardin','Jasper','Jefferson',\
         'Liberty','Newton','Orange'],
    'S':['Calhoun','De Witt','Goliad','Jackson','Lavaca','Victoria'],
    'T':['Jim Hogg','Webb','Zapata'],
    'U':['Aransas','Bee','Brooks','Duval','Jim Wells','Kenedy','Kleberg',\
         'Live Oak','McMullen','Nueces','Refugio','San Patricio'],
    'V':['Cameron','Hidalgo','Starr','Willacy']}
    
tsa_names = {'A':'Amarillo/Panhandle','B':'Lubbock/South Plains',\
             'C':'Wichita Falls/North Texas','D':'Abilene/Big Country',\
             'E':'Dallas/North-Central Texas','F':'Paris/Northeast Texas',\
             'G':'Tyler/Piney Woods','H':'Lufkin/Deep East Texas',\
             'I':'El Paso/Borderlands','J':'Midland/Permian Basin',\
             'K':'San Angelo/Concho Valley','L':'Killeen/Central Texas',\
             'M':'Waco/Heart of Texas','N':'Bryan/Brazos Valley',\
             'O':'Austin/Capital Area','P':'San Antonio/Southwest Texas',
             'Q':'Houston/Southeast Texas',\
             'R':'Galveston/East Texas Gulf Coast',\
             'S':'Victoria/Golden Crescent','T':'Laredo/Seven Flags',\
             'U':'Corpus Christi/Coastal Bend',\
             'V':'McAllen/Lower Rio Grande Valley'}

case_trend = [['TSA','new cases','one week ago','two weeks ago','30 days ago',\
        '7-day trend','14-day trend','30-day trend']]    
    
# case plots
for region in regions:
    if debug_mode and not plot_cases:
        break
    print('Cases: Region %s' % region)
    region_cases_df = cases_df[cases_df.index.isin(tsa_counties[region])].sum()
    fig,ax = plt.subplots(figsize=(20,8))
    plt.bar(region_cases_df.index[-180:],region_cases_df[-180:],color='blue',\
            label='Daily Reported Cases')
    plt.plot(region_cases_df.index[-180:],\
             region_cases_df[-180:].rolling(window=7).mean(),color='red',\
             label='7-Day Moving Average',marker='o',markevery=[-1])
        
    plt.text(region_cases_df.index[-1],\
        region_cases_df.rolling(window=7).mean()[-1],'{0:,.0f}'.\
        format(region_cases_df.rolling(window=7).mean()[-1]),color='red')
        
    plt.grid(which='major',axis='x',linestyle='-',linewidth=2)
    plt.grid(which='minor',axis='x',linestyle='--')
    plt.grid(which='major',axis='y',linestyle='-')
    
    plt.title('Last 180 Days of COVID-19 Cases in Trauma Service Region %s (%s)'\
              % (region,tsa_names[region]),fontsize=16)
    plt.xlabel('Report Date',fontsize=16)
    plt.ylabel('Cases',fontsize=16)
    
    plt.gca().set_ylim(bottom=0)
    
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,p: format(int(x),',')))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=(1,8,15,22)))
    
    plt.legend(loc='upper right',fontsize=14)
    
    plt.savefig('images/cases_%s.png' % region.lower(),bbox_inches='tight')
    plt.close('all')
    
    casetoday,caseweek,case2week,casemonth,casechange,case2change,case30change = \
        trends(region_cases_df.rolling(window=7).mean())
        
    case_trend.append(['%s' % region,'%0.0f' % casetoday,'%0.0f' % caseweek,\
                       '%0.0f' % case2week,'%0.0f' % casemonth,\
                       '%0.3f' % casechange,'%0.3f' % case2change,\
                        '%0.3f' % case30change])
        
makeCSV('case_trend.csv',case_trend)

# death plots

death_trend = [['TSA','new cases','one week ago','two weeks ago','30 days ago',\
        '7-day trend','14-day trend','30-day trend']] 
    
for region in regions:
    if debug_mode and not plot_deaths:
        break
    print('Deaths: Region %s' % region)
    region_deaths_df = daily_deaths_df[daily_deaths_df.index.isin(x.upper() for x in tsa_counties[region])].sum()
    fig,ax = plt.subplots(figsize=(20,8))
    plt.bar(region_deaths_df.index[-180:],region_deaths_df[-180:],color='blue',\
            label='Daily Reported Deaths')
    plt.plot(region_deaths_df.index[-180:],\
             region_deaths_df[-180:].rolling(window=7).mean(),color='red',\
             label='7-Day Moving Average',marker='o',markevery=[-1])
    
    plt.text(region_deaths_df.index[-1],\
        region_deaths_df.rolling(window=7).mean()[-1],'{0:,.0f}'.\
        format(region_deaths_df.rolling(window=7).mean()[-1]),color='red')    
        
    plt.grid(which='major',axis='x',linestyle='-',linewidth=2)
    plt.grid(which='minor',axis='x',linestyle='--')
    plt.grid(which='major',axis='y',linestyle='-')
    
    plt.title('Last 180 Days of COVID-19 Deaths in Trauma Service Region %s (%s)'\
              % (region,tsa_names[region]),fontsize=16)
    plt.xlabel('Report Date',fontsize=16)
    plt.ylabel('Deaths',fontsize=16)
    
    plt.gca().set_ylim(bottom=0)
    
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,p: format(int(x),',')))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=(1,8,15,22)))
    
    plt.legend(loc='upper right',fontsize=14)
    
    plt.savefig('images/deaths_%s.png' % region.lower(),bbox_inches='tight')
    plt.close('all')
    
    deathstoday,deathsweek,deaths2week,deathsmonth,deathschange,deaths2change,\
        deaths30change = trends(region_deaths_df.rolling(window=7).mean())
        
    death_trend.append(['%s' % region,'%0.0f' % deathstoday,'%0.0f' % deathsweek,\
                       '%0.0f' % deaths2week,'%0.0f' % deathsmonth,\
                       '%0.3f' % deathschange,'%0.3f' % deaths2change,\
                        '%0.3f' % deaths30change])
        
makeCSV('death_trend.csv',death_trend)

# positivity rate

pos_trend = [['TSA','new cases','one week ago','two weeks ago','30 days ago',\
        '7-day trend','14-day trend','30-day trend']] 

for region in regions:
    if debug_mode and not plot_positivity:
        break
    print('Positivity: Region %s' % region)
    region_cases_df = cases_df[cases_df.index.isin(tsa_counties[region])].sum()
    region_tests_df = daily_testing_df[daily_testing_df.index.isin(tsa_counties[region])].sum()
    
    region_cases_running_df = region_cases_df.rolling(window=7).sum()
    region_tests_running_df = region_tests_df.rolling(window=7).sum()
    
    region_positivity_df = pd.merge(pd.Series(region_cases_running_df,\
        name='cases'),pd.Series(region_tests_running_df,name='tests'),\
        left_index=True,right_index=True)
        
    region_positivity_df['positivity'] = region_positivity_df['cases'] / \
        region_positivity_df['tests']
        
    fig,ax = plt.subplots(figsize=(20,8))
    plt.bar(region_positivity_df.index[-180:],\
            region_positivity_df['positivity'][-180:]*100.0,color='blue',\
            label='7-Day Total Positivity')
    plt.plot(region_positivity_df.index[-180:],\
             region_positivity_df['positivity'][-180:].rolling(window=7).mean()*100.0,\
             color='red',label='7-Day Positivity (smoothed)',marker='o',markevery=[-1])
        
    plt.text(region_positivity_df.index[-1],\
        region_positivity_df['positivity'].rolling(window=7).mean()[-1]*100.0+1,'{0:,.1f}%'.\
        format(region_positivity_df['positivity'].rolling(window=7).mean()[-1]*100.0),color='red')
        
    plt.grid(which='major',axis='x',linestyle='-',linewidth=2)
    plt.grid(which='minor',axis='x',linestyle='--')
    plt.grid(which='major',axis='y',linestyle='-')
    
    plt.title('Last 180 Days of COVID-19 Test Positivity in Trauma Service Region %s (%s)'\
              % (region,tsa_names[region]),fontsize=16)
    plt.xlabel('Report Date',fontsize=16)
    plt.ylabel('Positivity Rate',fontsize=16)
    
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=(1,8,15,22)))
    
    plt.ylim([0,50])
    
    plt.savefig('images/positivity_%s.png' % region.lower(),bbox_inches='tight')
    plt.close('all')
    
    postoday,posweek,pos2week,posmonth,poschange,pos2change,\
        pos30change = trends(region_positivity_df['positivity'].rolling(window=7).mean())
        
    pos_trend.append(['%s' % region,'%0.0f' % postoday,'%0.0f' % posweek,\
                       '%0.0f' % pos2week,'%0.0f' % posmonth,\
                       '%0.3f' % poschange,'%0.3f' % pos2change,\
                        '%0.3f' % pos30change])
        
makeCSV('pos_trend.csv',pos_trend)

# hospitalization plots

hosp_trend = [['TSA','new cases','one week ago','two weeks ago','30 days ago',\
        '7-day trend','14-day trend','30-day trend']] 

for region in regions:
    if debug_mode and not plot_hospital:
        break
    print('Hospitalizations: Region %s' % region)
    fig,ax = plt.subplots(figsize=(20,8))
    plt.bar(hospital_df.columns[-180:],hospital_df.loc[region][-180:]*100.0,\
            color='blue',label='% of hospital capacity occupied by COVID patients')
    plt.plot(hospital_df.columns[-180:],\
             hospital_df.loc[region][-180:].rolling(window=7).mean()*100.0,color='red',\
             label='7-Day Moving Average',marker='o',markevery=[-1])

    plt.text(hospital_df.columns[-1],\
        hospital_df.loc[region][-180:].rolling(window=7).mean()[-1]*100.0+1,'{0:,.1f}%'.\
        format(hospital_df.loc[region][-180:].rolling(window=7).mean()[-1]*100.0),color='red')
        

    plt.grid(which='major',axis='x',linestyle='-',linewidth=2)
    plt.grid(which='minor',axis='x',linestyle='--')
    plt.grid(which='major',axis='y',linestyle='-')
    
    plt.title('Total Hospital Capacity Occupied by COVID Patients in Trauma Service Region %s (%s)'\
              % (region,tsa_names[region]),fontsize=16)
    plt.xlabel('Report Date',fontsize=16)
    plt.ylabel('Percent Utilization (%)',fontsize=16)
    
    plt.ylim([0,50])
    
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=(1,8,15,22)))
    
    plt.legend(loc='upper right',fontsize=14)
    
    plt.savefig('images/hospital_%s.png' % region.lower(),bbox_inches='tight')
    plt.close('all')
    
    hosptoday,hospweek,hosp2week,hospmonth,hospchange,hosp2change,\
        hosp30change = trends(hospital_df.loc[region].rolling(window=7).mean())
        
    hosp_trend.append(['%s' % region,'%0.0f' % hosptoday,'%0.0f' % hospweek,\
                       '%0.0f' % hosp2week,'%0.0f' % hospmonth,\
                       '%0.3f' % hospchange,'%0.3f' % hosp2change,\
                        '%0.3f' % hosp30change])
        
makeCSV('hosp_trend.csv',hosp_trend)

# regional transmission data

# get population data
population_df = pd.read_excel('county-populations.xlsx',index_col='County',dtype={'Population':float})
region_populations = {}
region_cases_per100k = {}
case_value = {}
positivity_value = {}
transmission = {}
for region in regions:
    region_populations[region] = float(population_df[population_df.index.isin(tsa_counties[region])].sum())
    # compute the cases per 100K by region
    region_cases_per100k = cases_df[cases_df.index.isin(tsa_counties[region])].sum() / (region_populations[region]/100000.0)
    case_value[region] = region_cases_per100k.rolling(window=7).mean()[-1]
    # compute the 7-day test positivity
    region_cases_df = cases_df[cases_df.index.isin(tsa_counties[region])].sum()
    region_tests_df = daily_testing_df[daily_testing_df.index.isin(tsa_counties[region])].sum()
    
    region_cases_running_df = region_cases_df.rolling(window=7).sum()
    region_tests_running_df = region_tests_df.rolling(window=7).sum()
    
    region_positivity_df = pd.merge(pd.Series(region_cases_running_df,\
    name='cases'),pd.Series(region_tests_running_df,name='tests'),\
    left_index=True,right_index=True)
        
    region_positivity_df['positivity'] = region_positivity_df['cases'] / \
        region_positivity_df['tests']
    
    positivity_value[region] = region_positivity_df['positivity'].rolling(window=7).mean()[-1]
    
    if case_value[region] >= 100 or positivity_value[region] >= 0.1:
        transmission[region] = 'High'
    elif case_value[region] >= 50 or positivity_value[region] >= 0.08:
        transmission[region] = 'Substantial'
    elif case_value[region] >= 10 or positivity_value[region] >= 0.05:
        transmission[region] = 'Moderate'
    elif case_value[region] < 10 and positivity_value[region] < 0.05:
        transmission[region] = 'Low'
    else:
        transmission[region] = 'Not enough data'
        
transmission_df = pd.DataFrame.from_dict(transmission,orient='index')

# create regional heatmaps
plt.close('all')
fig,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(15,12))
# cases per 100K
ax1 = sns.heatmap(pd.DataFrame.from_dict(case_value,orient='index'),vmin=0,\
            vmax=200,cmap='magma',annot=True,fmt='.0f',linewidths=1,\
            linecolor='black',xticklabels=False,cbar=False,ax=ax1,\
            annot_kws={'fontsize':20})
ax1.set_yticklabels(pd.DataFrame.from_dict(case_value,orient='index').index,\
            rotation=0)

# positivity
ax2 = sns.heatmap(pd.DataFrame.from_dict(positivity_value,orient='index')*100.0,\
        vmin=0,vmax=50,cmap='magma',annot=True,fmt='.1f',linewidths=1,\
        linecolor='black',xticklabels=False,\
        cbar=False,ax=ax2,annot_kws={'fontsize':20})
for t in ax2.texts: t.set_text(t.get_text() + '%')
ax2.set_yticklabels(pd.DataFrame.from_dict(positivity_value,orient='index').index,\
            rotation=0)

# hospitalizations
hosp_recent = pd.DataFrame({'Hosp':hospital_df.rolling(window=7,\
        min_periods=1).mean().iloc[:,-1]},index=hospital_df.index)
ax3 = sns.heatmap(hosp_recent*100.0,vmin=0,vmax=30,cmap='magma',annot=True,\
        fmt='.1f',linewidths=1,linecolor='black',xticklabels=False,\
        cbar=False,ax=ax3,annot_kws={'fontsize':20})
for t in ax3.texts: t.set_text(t.get_text() + '%')
ax3.set_yticklabels(hospital_df.index,rotation=0)
ax3.set(ylabel=None)

# transmission level
value_to_int = {'Low':0,'Moderate':1,'Substantial':2,'High':3}
n = len(value_to_int)
colors = ['#1d8aff','#fff70e','#ff7134','#ff0000']
cmap = sns.color_palette(colors,n)

ax4 = sns.heatmap(transmission_df.replace(value_to_int),cmap=cmap,vmin=0,\
        vmax=3,annot=transmission_df,fmt='',linewidth=1,linecolor='black',\
        xticklabels=False,cbar=False,ax=ax4,annot_kws={'fontsize':20})
ax4.set_yticklabels(transmission_df.index,rotation=0)
    
# uncomment the code block below if you want a colorbar for transmission
'''
colorbar = ax4.collections[0].colorbar
r = colorbar.vmax - colorbar.vmin
colorbar.set_ticks([colorbar.vmin + r/n * (0.5 + i) for i in range(n)])
colorbar.set_ticklabels(list(value_to_int.keys()))
'''

# y-tick font sizes
ax1.set_yticklabels(ax1.get_yticklabels(),size=16)
ax2.set_yticklabels(ax2.get_yticklabels(),size=16)
ax3.set_yticklabels(ax3.get_yticklabels(),size=16)
ax4.set_yticklabels(ax4.get_yticklabels(),size=16)

# plot titles
ax1.set_title('Cases per 100K',size=20)
ax2.set_title('Positivity Rate',size=20)
ax3.set_title('Hospital Utilization',size=20)
ax4.set_title('Transmission Level',size=20)

fig.tight_layout()
plt.savefig('images/heatmap.png',bbox_inches='tight')
    
print('Done.')