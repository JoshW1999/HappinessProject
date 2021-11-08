import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
from matplotlib.gridspec import GridSpec


'''

	Initialization and Pre-Processing

'''

# World Happiness Survey  
wh_17 = pd.read_csv('2017.csv')
wh_17.set_index('Country', inplace=True)

# Life Expectancy by Country
life_expectancy = pd.read_csv('LifeExpectancyByCountry.csv')
life_expectancy.set_index('Country Name', inplace=True)

# General Country statistics
country_stats = pd.read_csv('countries_of_the_world.csv', decimal=',') 

# Truncating extra space following country name
country_stats['Country'] = country_stats['Country'].apply(lambda x: x[:-1])
country_stats.set_index('Country', inplace=True)

# Removing any items that are not in other dataset
country_stats = country_stats[country_stats.index.isin(wh_17.index)]
wh_17 = wh_17[wh_17.index.isin(country_stats.index)]
life_expectancy = life_expectancy[life_expectancy.index.isin(country_stats.index)]

# Sorting by Index (Country names)
country_stats.sort_index(inplace=True)
wh_17.sort_index(inplace=True)
life_expectancy.sort_index(inplace=True)


#print(country_stats.columns)
#print(wh_17.columns)

plt.style.use('fivethirtyeight')

print(np.mean(wh_17['Happiness.Score']))

'''
	
	Phones and Happiness Score

'''
h_scores = wh_17['Happiness.Score']
phones = country_stats['Phones (per 1000)']

# Truncating Null Values
h_scores = h_scores[phones.notnull()]
phones = phones[phones.notnull()]

### Plotting ###
fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
fig1.set_size_inches(10, 10)

# ax1 is a scatter plot
ax1.scatter(phones.to_numpy(), h_scores.to_numpy())

# Line of best fit
m, b = np.polyfit(phones, h_scores, 1)
ax1.plot(phones.to_numpy(), m*phones.to_numpy() + b, 'r-', linewidth=1)

# Correlation
hscore_phones_corr = np.corrcoef(phones.to_numpy(), h_scores.to_numpy())
ax1.annotate("Correlation Coefficient: " + str(np.round(hscore_phones_corr[0][1],3)), (600,3), fontsize=14)

# Setting Labels
ax1.set_title('Prevalence of Phones vs. Happiness Score')
ax1.set_xlabel('Phones (per 1000)')
ax1.set_ylabel('Happiness Score')


# ax2 is a histogram

# Defining Bin Ranges
phones_range = np.linspace(0, 1000, 6)

counts, bins, patches = ax2.hist(phones, bins=phones_range, edgecolor='k')
ax2.set_xticks(bins)

# Calculate the center of each bin
bin_centers = 0.5 * np.diff(bins) + bins[:-1]

# Average Happiness Score per Bin
bin_avgs = [np.mean(h_scores[(phones > phones_range[i-1]) & (phones <= phones_range[i])]) for i in range(1, len(phones_range))]


# Annotations for counts and avg happiness score per bin
for bin_avg, count, x in zip(bin_avgs, counts, bin_centers):
	# Label Raw Count
	ax2.annotate(str(int(count)), xy=(x,0), xycoords=('data', 'axes fraction'), xytext=(0,-18), textcoords='offset points', va='top', ha='center')

	# Average Happiness Score per Bin
	ax2.annotate('%.2f' % bin_avg, xy=(x,0), xycoords=('data', 'axes fraction'), xytext=(0, -32), textcoords='offset points', va='top', ha='center')


# Setting Labels
ax2.set_title('Histogram of Phones (per 1000)', y=0.9)
ax2.set_ylabel('Frequency')
ax2.set_xlabel('Average Happiness Score (per bin)')

# Adjusting Padding
ax2.xaxis.labelpad = 30
plt.subplots_adjust(bottom=0.15)

plt.show()

'''

	Literacy vs. Happiness Score

'''
print(country_stats.columns)
literacy_rate = country_stats['Literacy (%)']
h_scores = wh_17['Happiness.Score']

# Only considering non-null values
h_scores = h_scores[literacy_rate.notnull()]
literacy_rate = literacy_rate[literacy_rate.notnull()]

fig2, (ax3, ax4) = plt.subplots(nrows=2, ncols=1)
fig2.set_size_inches(10, 10)

# ax1 is a scatter plot
ax3.scatter(literacy_rate.to_numpy(), h_scores.to_numpy())

# Line of best fit
m, b = np.polyfit(literacy_rate, h_scores, 1)
ax3.plot(literacy_rate.to_numpy(), m*literacy_rate.to_numpy() + b, 'r-', linewidth=1)

# Correlation
hscore_lit_corr = np.corrcoef(literacy_rate.to_numpy(), h_scores.to_numpy())
ax3.annotate("Correlation Coefficient: " + str(np.round(hscore_lit_corr[0][1],3)), (20,7), fontsize=14)

# Setting Labels
ax3.set_title('Literacy Rate vs. Happiness Score')
ax3.set_xlabel('Literacy Rate (%)')
ax3.set_ylabel('Happiness Score')

rates = [0,25,50,75,100]
literacy_discretized = [len(h_scores[(literacy_rate >= rates[i-1]) & (literacy_rate < rates[i])]) for i in range(1, len(rates))]
literacy_by_happiness = ["%.2f" % np.mean(h_scores[(literacy_rate >= rates[i-1]) & (literacy_rate < rates[i])]) for i in range(1, len(rates))]

#print(zip(rates[1:], literacy_by_happiness))

wedges, _, _ = ax4.pie(literacy_discretized, labels=['<25%', '25-50%', '50-75%', '>75%'], autopct='%.2f%%', explode=np.repeat(0.1, 4))
ax4.set_title('Literacy Rates and Happiness Score', y=-.2, fontsize=16)

ax4.legend(wedges, literacy_by_happiness, loc='upper left', bbox_to_anchor=[-.8,1] , title='Avg. Happiness Scores')

plt.show()

'''

	GDP vs. Happiness Score

'''


gdp = country_stats['GDP ($ per capita)']
h_scores = wh_17['Happiness.Score']

# Only considering non-null values
h_scores = h_scores[gdp.notnull()]
gdp = gdp[gdp.notnull()]

fig3, (ax5, ax6) = plt.subplots(2,1)
fig3.set_size_inches(10,10)

# ax1 is a scatter plot
ax5.scatter(gdp.to_numpy(), h_scores.to_numpy())

# Line of best fit
m, b = np.polyfit(gdp, h_scores, 1)
ax5.plot(gdp.to_numpy(), m*gdp.to_numpy() + b, 'r-', linewidth=1, label='Line of best fit')

# Plotting International Poverty Line
# Source: https://www.worldvision.org/sponsorship-news-stories/global-poverty-facts
ex_pov_line = 1.90 * 365
low_mid_pov_line = 3.20 * 365
up_mid_pov_line = 5.50 * 365
up_pov_line = 21.70 * 365

#ax5.axvline(ex_pov_line, color='g', linestyle='--', linewidth=1, label='Extreme Poverty Line')

ax5.axvspan(0, ex_pov_line, color='y', alpha=0.25, label='Extreme Poverty Line')
#ax5.axvspan(ex_pov_line, low_mid_pov_line, color='y',alpha=0.25, label='Lower-Mid Poverty Line')
#ax5.axvspan(low_mid_pov_line, up_mid_pov_line, color='b',alpha=0.25, label='Upper-Mid Poverty Line')
#ax5.axvspan(up_mid_pov_line, up_pov_line, color='g', alpha=0.25, label='Upper Poverty Line')


# Correlation
hscore_gdp_corr = np.corrcoef(gdp.to_numpy(), h_scores.to_numpy())
ax5.annotate("Correlation Coefficient: " + str(np.round(hscore_gdp_corr[0][1],3)), (35000,6), fontsize=14)

# Setting Labels
ax5.set_title('GDP vs. Happiness Score')
ax5.set_xlabel('GDP ($ per capita)')
ax5.set_ylabel('Happiness Score')
ax5.legend()

# Pie Chart
gdp_bins = np.linspace(0,max(gdp)+1, 6)

gdp_discretized = [len(h_scores[(gdp >= gdp_bins[i-1]) & (gdp < gdp_bins[i])]) for i in range(1, len(gdp_bins))]
gdp_by_happiness = ["%.2f" % np.mean(h_scores[(gdp >= gdp_bins[i-1]) & (gdp < gdp_bins[i])]) for i in range(1, len(gdp_bins))]

labels = ['\${:.0f}'.format(gdp_bins[i-1]) + '-' + '\${:.0f}'.format(gdp_bins[i]) for i in range(1, len(gdp_bins))] # Escape Character Required for matplotlib display

wedges, _ = ax6.pie(gdp_discretized, labels=labels, explode = np.repeat(0.1,5) + np.array([0,0,0,0.4,0.4]))
ax6.set_title('GDP per capita and Happiness Score', y=-.2, fontsize=16)

ax6.legend(wedges, gdp_by_happiness, loc='upper left', bbox_to_anchor=[-.8,1] , title='Avg. Happiness Scores')


plt.show()


'''

	Birthrate vs Happiness Score vs Life Expectancy

'''
birth_rate = country_stats['Birthrate']
h_scores = wh_17['Happiness.Score']
life_expectancy_2017 = life_expectancy['2017']

# life_expectancy_2017 is smaller than the other two lists; adjusting accordingly: 
birth_rate = birth_rate[birth_rate.index.isin(life_expectancy_2017.index)]
h_scores = h_scores[h_scores.index.isin(life_expectancy_2017.index)]

# Only considering non-null values
h_scores = h_scores[birth_rate.notnull()]
life_expectancy_2017 = life_expectancy_2017[birth_rate.notnull()]
birth_rate = birth_rate[birth_rate.notnull()]

fig4, (ax7, ax8) = plt.subplots(2,1)
fig4.set_size_inches(10,10)
fig4.tight_layout()
# ax7 is a scatter plot

# Color Map for showing life-expectancy
cm = plt.get_cmap('jet')
ax7.scatter(birth_rate.to_numpy(), h_scores.to_numpy(), c=life_expectancy_2017.to_numpy(), cmap=cm)

# Color Bar for reference
cbar = plt.cm.ScalarMappable(cmap='jet')
cbar.set_array(life_expectancy_2017.to_numpy())


# Line of best fit
m, b = np.polyfit(birth_rate, h_scores, 1)
ax7.plot(birth_rate.to_numpy(), m*birth_rate.to_numpy() + b, 'r-', linewidth=1, label='Line of best fit')


# Correlation
h_score_br_le_corr = np.corrcoef([birth_rate.to_numpy(), h_scores.to_numpy(), life_expectancy_2017.to_numpy()])
#print(h_score_br_le_corr)


# Setting Labels
ax7.set_title('Birthrate vs. Happiness Score')
ax7.set_xlabel('Birthrate (per 1000)')
ax7.set_ylabel('Happiness Score')
ax7.legend()

# Plotting Correlation Matrix
axes = ['Birthrate', 'Happiness Score', 'Life Expectancy']

ax8.matshow(h_score_br_le_corr, cmap='jet')

ax8.set_title('Correlation Matrix')

ax8.set_xticks(range(len(axes)))
ax8.set_xticklabels(axes, fontsize=10)

ax8.set_yticks(range(len(axes)))
ax8.set_yticklabels(axes, fontsize=10)

cbar_2 = plt.cm.ScalarMappable(cmap='jet')
cbar_2.set_array(h_score_br_le_corr)

plt.colorbar(cbar, ax=ax7, label='Life Expectancy')
plt.colorbar(cbar_2, ax=ax8, label='Correlation')
plt.subplots_adjust(top=0.95,hspace=0.5)
plt.show()



'''

	Agriculture vs. Happiness Score vs. GDP 

'''
h_scores = wh_17['Happiness.Score']
agri = country_stats['Agriculture'] # % Agriculture Composition of GDP
industry = country_stats['Industry'] # % Industrial Composition of GDP
service = country_stats['Service'] # % Service Composition of GDP

gdp = country_stats['GDP ($ per capita)']

h_scores = h_scores[(agri.notnull()) & (gdp.notnull())]
agri = agri[(agri.notnull()) & (gdp.notnull())]
gdp = gdp[(agri.notnull()) & (gdp.notnull())]




#fig5, (ax9, ax10, ax11, ax12) = plt.subplots(4,1)
fig5 = plt.figure(constrained_layout=True)
fig5.set_size_inches(20,14)
#fig5.tight_layout()

gs = GridSpec(4,3, figure=fig5)

# Stacked Bar Graph of Nation's GDP sector Composition
ax9 = fig5.add_subplot(gs[0,:])

# Scatter Plots for each branch vs Happiness Score w/GDP

# Agriculture
ax10 = fig5.add_subplot(gs[1,0])
ax11 = fig5.add_subplot(gs[1,1])
ax12 = fig5.add_subplot(gs[1,2])

# Industry
ax13 = fig5.add_subplot(gs[2,0])
ax14 = fig5.add_subplot(gs[2,1])
ax15 = fig5.add_subplot(gs[2,2])

# Service
ax16 = fig5.add_subplot(gs[3,0])
ax17 = fig5.add_subplot(gs[3,1])
ax18 = fig5.add_subplot(gs[3,2])

# Stacked Bar Graph of Nation's GDP sector Composition 
ax9.bar(country_stats.index, agri.to_numpy(), width=0.3, label='Agriculture')
ax9.bar(country_stats.index, industry.to_numpy(), bottom=agri.to_numpy(), width=0.3, label='Industry')
ax9.bar(country_stats.index, service.to_numpy(), width=0.3, bottom=agri.to_numpy()+industry.to_numpy(), label='Service')

ax9.set_title('Economic Sector Composition to GDP')
ax9.set_xticklabels(country_stats.index, fontsize=6, rotation=90)


# Color map for displaying life expectancy
cm = plt.get_cmap('jet')

# Color Bar for reference
cbar = plt.cm.ScalarMappable(cmap='Greens')
cbar.set_array(gdp.to_numpy())

# Agriculture vs Happiness Score Scatter Plot 
ax10.scatter(agri.to_numpy(), h_scores.to_numpy(), c=gdp.to_numpy(), cmap='Greens')

# Line of best fit
m, b = np.polyfit(agri, h_scores, 1)
ax10.plot(agri.to_numpy(), m*agri.to_numpy() + b, 'r-', linewidth=1, label='Line of best fit')

ax10.set_xlabel('Agriculture Sector Proportion to GDP', fontsize=8)
ax10.set_ylabel('Happiness Score')

# Correlation
h_score_agri_gdp_corr = np.corrcoef([agri.to_numpy(), h_scores.to_numpy(), gdp.to_numpy()])



# Industry vs Happiness Score Scatter Plot
ax11.scatter(industry.to_numpy(), h_scores.to_numpy(), c=gdp.to_numpy(), cmap='Greens')

ax11.set_xlabel('Industry Sector Proportion to GDP', fontsize=8)
#ax10.set_ylabel('Happiness Score')

# Line of best fit
m, b = np.polyfit(industry, h_scores, 1)
ax11.plot(industry.to_numpy(), m*industry.to_numpy() + b, 'r-', linewidth=1, label='Line of best fit')

# Correlation
h_score_industry_gdp_corr = np.corrcoef([industry.to_numpy(), h_scores.to_numpy(), gdp.to_numpy()])



# Service vs Happiness Score Scatter Plot
ax12.scatter(service.to_numpy(), h_scores.to_numpy(), c=gdp.to_numpy(), cmap='Greens')

ax12.set_xlabel('Service Sector Proportion to GDP', fontsize=8)
#ax10.set_ylabel('Happiness Score')

# Line of best fit
m, b = np.polyfit(service, h_scores, 1)
ax12.plot(service.to_numpy(), m*service.to_numpy() + b, 'r-', linewidth=1, label='Line of best fit')

# Correlation
h_score_service_gdp_corr = np.corrcoef([service.to_numpy(), h_scores.to_numpy(), gdp.to_numpy()])



# Plotting Correlation Matrices
axes_a = ['Agriculture', 'Happiness Score', 'GDP']
axes_i = ['Industry', 'Happiness Score', 'GDP']
axes_s = ['Service', 'Happiness Score', 'GDP']

# Agriculture Correlation Matrix
ax13.matshow(h_score_agri_gdp_corr, cmap='jet', vmin=-1.0, vmax=1.0)
#ax13.set_title('Correlation Matrix')

ax13.set_xticks(range(len(axes_a)))
ax13.set_xticklabels(axes_a, fontsize=5)

ax13.set_yticks(range(len(axes_a)))
ax13.set_yticklabels(axes_a, fontsize=6)


# Industry Correlation Matrix
ax14.matshow(h_score_industry_gdp_corr, cmap='jet', vmin=-1.0, vmax=1.0)
#ax14.set_title('Correlation Matrix')

ax14.set_xticks(range(len(axes_i)))
ax14.set_xticklabels(axes_i, fontsize=5)

ax14.set_yticks(range(len(axes_i)))
ax14.set_yticklabels(axes_i, fontsize=6)


# Service Correlation Matrix
ax15.matshow(h_score_service_gdp_corr, cmap='jet', vmin=-1.0, vmax=1.0)
#ax15.set_title('Correlation Matrix')

ax15.set_xticks(range(len(axes_s)))
ax15.set_xticklabels(axes_s, fontsize=5)

ax15.set_yticks(range(len(axes_s)))
ax15.set_yticklabels(axes_s, fontsize=6)

# Color bars for respective correlation matrices
cbar_2 = plt.cm.ScalarMappable(cmap='jet')
cbar_2.set_array(np.concatenate((h_score_agri_gdp_corr, h_score_industry_gdp_corr, h_score_service_gdp_corr)))



# Box Plot to showcase variation between GDP Composition Branches
ax16.boxplot([agri, industry, service], labels = ['Agriculture', 'Industry', 'Service'])
ax16.set_ylabel('Economy Dependency')

# Counts for nations with respective GDP Composition Branches
agri_based = len(country_stats[(country_stats['Agriculture'] > country_stats['Industry']) & (country_stats['Agriculture'] > country_stats['Service'])])
industry_based = len(country_stats[(country_stats['Industry'] > country_stats['Agriculture']) & (country_stats['Industry'] > country_stats['Service'])])
service_based = len(country_stats[(country_stats['Service'] > country_stats['Agriculture']) & (country_stats['Service'] > country_stats['Industry'])])

# Average GDP for respective largest GDP Branch
gdp_agri_based = np.mean(country_stats['GDP ($ per capita)'][(country_stats['Agriculture'] > country_stats['Industry']) & (country_stats['Agriculture'] > country_stats['Service'])])
gdp_industry_based = np.mean(country_stats['GDP ($ per capita)'][(country_stats['Industry'] > country_stats['Agriculture']) & (country_stats['Industry'] > country_stats['Service'])])
gdp_service_based = np.mean(country_stats['GDP ($ per capita)'][(country_stats['Service'] > country_stats['Agriculture']) & (country_stats['Service'] > country_stats['Industry'])])


# Pie Graph displaying Counts of Nations' largest GDP Branch
wedges, text, autotexts = ax17.pie([agri_based,industry_based,service_based], labels=['Agriculture', 'Industry', 'Service'], explode=np.repeat(0.1, 3), autopct='%.2f%%', textprops={'fontsize':8})
ax17.set_title('Nations\' largest economic branch')

# Getting average happiness score per nations' largest economic sector
happiness_by_gdp_comp = ["%.2f" % np.mean(h_scores[(country_stats['Agriculture'] > country_stats['Industry']) & (country_stats['Agriculture'] > country_stats['Service'])]),
	"%.2f" % np.mean(h_scores[(country_stats['Industry'] > country_stats['Agriculture']) & (country_stats['Industry'] > country_stats['Service'])]),
		"%.2f" % np.mean(h_scores[(country_stats['Service'] > country_stats['Agriculture']) & (country_stats['Service'] > country_stats['Industry'])])]

ax17.legend(wedges, happiness_by_gdp_comp, loc='upper left', bbox_to_anchor=[0,0] , title='Avg. Happiness Scores', prop={'size':10}, fontsize=8)

# Bar Plot displaying Nations' average GDP based on their largest GDP composition branch
ax18.barh(['Agriculture', 'Industry', 'Service'], [gdp_agri_based, gdp_industry_based, gdp_service_based])
ax18.set_title('Average GDP per Largest Economic Branch')
ax18.set_xlabel('GDP ($)')

# Display
ax9.legend(loc='best', bbox_to_anchor=[0.025,0,1,1], fontsize=8)
ax10.legend()
ax11.legend()
ax12.legend()
plt.colorbar(cbar, ax=[ax10,ax11,ax12], label='GDP per capita')
plt.colorbar(cbar_2, ax=[ax13,ax14,ax15], label='Correlation')

plt.show()


'''
	
	All correlations to Happiness Score

'''

fig6, ax19 = plt.subplots()
fig6.set_size_inches(12,10)

h_scores = [wh_17['Happiness.Score'][country_stats[col].notnull()] for col in country_stats.columns[1:]] # 1: to skip Regions Column (non-numeric)
stats = [country_stats[col][country_stats[col].notnull()] for col in country_stats.columns[1:]]


ax19.bar(country_stats.columns[1:], [np.corrcoef(h_scores[i], stats[i])[1][0] for i in range(len(h_scores))])

ax19.set_title('All Correlations w/ Happiness Score')

ax19.set_xticklabels(country_stats.columns[1:], rotation=90, fontsize=12)

ax19.set_ylabel('Strength of Correlation')



plt.subplots_adjust(bottom=0.3)
plt.show()


# Prevalence of Phones vs GDP

fig7, ax20 = plt.subplots()
fig7.set_size_inches(12,8)

phones = country_stats['Phones (per 1000)']
gdp = country_stats['GDP ($ per capita)']

phones = phones[(gdp.notnull()) & (phones.notnull())]
gdp = gdp[(gdp.notnull()) & (phones.notnull())]

# Correlation Coefficient
phones_gdp_corr = np.corrcoef(phones, gdp)
ax20.annotate("Correlation Coefficient: " + str(np.round(phones_gdp_corr[0][1],3)), (700,20000), fontsize=14)


# Line of best fit
m, b = np.polyfit(phones, gdp, 1)
ax20.plot(phones.to_numpy(), m*phones.to_numpy() + b, 'r-', linewidth=1, label='Line of best fit')


ax20.scatter(phones, gdp)

ax20.set_title('Prevalence of Phones vs. GDP')
ax20.set_xlabel('Phones (per 1000)')
ax20.set_ylabel('GDP ($ per capita)')

ax20.legend()

plt.show()


fig8, ax21 = plt.subplots()
fig8.set_size_inches(12,8)

migration = country_stats['Net migration']
gdp = country_stats['GDP ($ per capita)']

migration = migration[(migration.notnull()) & (gdp.notnull())]
gdp = gdp[(gdp.notnull()) & (migration.notnull())]

# Correlation
nm_gdp_corr = np.corrcoef(migration, gdp)

# Line of best fit
m, b = np.polyfit(migration, gdp, 1)
ax21.plot(migration.to_numpy(), m*migration.to_numpy() + b, 'r-', linewidth=1, label='Line of best fit')

# Displaying Correlation
ax21.annotate("Correlation Coefficient: " + str(np.round(nm_gdp_corr[0][1],3)), (15,40000), fontsize=14)


ax21.scatter(migration, gdp)
ax21.set_title('Net Migration vs. GDP')
ax21.set_xlabel('Net Migration')
ax21.set_ylabel('GDP ($ per capita)')
ax21.legend()

plt.show()

fig9, ax22 = plt.subplots()
fig9.set_size_inches(12,8)

gdp = country_stats['GDP ($ per capita)']
birth_rate = country_stats['Birthrate']

gdp = gdp[(gdp.notnull()) & birth_rate.notnull()]
birth_rate = birth_rate[(gdp.notnull()) & (birth_rate.notnull())]

# Correlation
gdp_br_corr = np.corrcoef(gdp, birth_rate)

# Line of best fit
m, b = np.polyfit(gdp, birth_rate, 1)
ax22.plot(gdp.to_numpy(), m*gdp.to_numpy() + b, 'r-', linewidth=1, label='Line of best fit')

# Displaying Correlation
ax22.annotate("Correlation Coefficient: " + str(np.round(gdp_br_corr[0][1],3)), (40000,20), fontsize=14)

# Scatter Plot
ax22.scatter(gdp, birth_rate)

ax22.set_title('GDP vs. Birthrate')
ax22.set_xlabel('GDP ($ per capita)')
ax22.set_ylabel('Birthrate (per 1000)')
ax22.legend()

plt.show()

fig1.savefig('phones_vs_hs')
fig2.savefig('Lit_vs_hs')
fig3.savefig('GDP_vs_hs')
fig4.savefig('br_vs_hs_vs_le')
fig5.savefig('econ_sector_vs_hs')
fig6.savefig('all_correlations')
fig7.savefig('phones_vs_gdp')
fig8.savefig('migration_vs_gdp')
fig9.savefig('gdp_vs_birthrate')


'''
	Happiness Score Trend

'''