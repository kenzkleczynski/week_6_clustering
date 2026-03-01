#%% import libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


#%% load code from Lab-Clustering.ipynb but shortened and condence to only include whats nescary for reflection

#i didnt comment on this beucase all my reasonings are in the lab notebook file

salaries = pd.read_csv('2025_salaries.csv', header=1, encoding='latin-1')
salaries.columns = ['Player', 'Team', 'Salary']
nba = pd.read_csv('nba_2025.txt', sep=',', encoding='latin-1')

merged = pd.merge(nba, salaries, on='Player', how='inner')
merged = merged.drop_duplicates(subset='Player', keep='first')
df = merged.drop(columns=['Team_x', 'Player-additional', 'Awards', 'Team_y'])
df['Salary'] = df['Salary'].str.replace('$', '').str.replace(',', '').astype(float)
df = df.dropna(subset=['Salary'])
df = df.drop(columns=['2P%','3P%','FT%'])

features = df[['PTS','AST']]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

model = KMeans(n_clusters=3, random_state=1)
model.fit(scaled)
df['cluster'] = model.labels_

#%% visualization

top_5 = df[df['cluster'] == 2].sort_values('Salary').head(5)
bottom_5 = df[df['cluster'] == 1].sort_values('Salary', ascending=False).head(5)

sns.scatterplot(data=df, x='PTS', y='AST', hue='Salary', style='cluster', palette='mako', alpha=0.5)
plt.scatter(top_5['PTS'], top_5['AST'], color='green', marker='*', s=200)
plt.scatter(bottom_5['PTS'], bottom_5['AST'], color='red', marker='x', s=200)
plt.legend(bbox_to_anchor=(1,1), loc=2)
plt.title('Player Value: Points vs Assists')
plt.xlabel('Total Points')
plt.ylabel('Total Assists')
plt.show()

# %% Reflection for non-tech audience

#Outline:

#What's the problem? — Wizards are bad, need better players, can't overspend
#As VP of anylitcs for the Wizards, I am helping head couter Mr. Rooney recuity the best players for the best value though data science. 
#the goal is to research for teh best skill level perfromers that are underpaid in order to rectuit them for cheap and build a playoff ready team
#without overpsending.

#What data did you use? — two datasets, stats and salaries, merged them together
# To compelte this project I had two intially data sets. One data set contrin infmoation of every NBA player, their team, and theie salary. Another data set listed every NBA players player inforation adn performacen stats.
# I merged these data sets by the player name throuhg an inner merge that only kept players that were in both data sets, this way I had a complete data set with all the information I needed to do my analysis.
# After meiging the data sets I cleaned the data by dropping duplicates after noting a pattern in teh teams column. When a player tranfered teams and had two teamsn and stats assocaited with them there was a seperate row that listen them as "2TM" and added ther stats to get a total.
#Additonally, i dropped other duplicate rows, columns with majority mssiing data, and some rows with missiing salary data. I also dropped the percentage columns because they had a lot of missing data.


#How does clustering work? — explain it simply, grouping similar players together without knowing the answer ahead of time
#in order to see patterns in this unsupervised data, I used a metho called k-means clustering. This method groups the players together
# by there similarites in traits from the chosen features. While you dont nedecaly ned to know that math behind it, a genrel thing to keep in mind
# is that its caludated on ditance and averages of the points. This is why it is importnat to scale the data to be on one univeral scale, 0-1, so that the 
# distances are not caualuced on skewed scales that may interfiew with the way distances between points are portrayed. 

#Why PTS and AST? — correlated most with salary, good indicators of overall performance
# In order for my cluserting to be shown I have to choose two features to plot on the x and y axis of a scatterplot that will best show what it means to be a player.
# i wanted to choose the most optimal feature that have a highcoorelation so that cluster groups can be distinclvly define. As well thought there needs to be a high varience, which means the data in teh feature is wide and spans over alot of numbers so we can see change.
#inorder to calcauted the varince of imporntat feauterd i defined like points scored, minutes played, assitsts, and a few more columns. I think plotts the corelation on a coorelation matrix 
# in order to see the feathers corelaiton with each other but also with salary.
# i found that points and assists had the highest correlation with salary, and also had a high varience, so I chose those two features to plot on the x and y axis of my scatterplot.

#What did the clusters show? — describe the three groups in plain english, best/mid/worst performers
# my target vriable, the thing i am trying to draw insights from, is salary. 
# so to find the best players I changed the color, hue, of the points to represent the salres they make. The darker the color the cheaper they are adn the more teal
#they getthe more expesive they are. The point is to find the dark color points that are in the top right side of the graph. This shows the players with high permoring stats, but are not getting paid alot.
# i clusted my data into 3 differet cluster. Cluster group 0 its the worst performing players, they have low points and low assists. Cluster group 1 is the mid performing players, they have a wide range of points and assists but are generally in the middle. Cluster group 2 is the best performing players, they have high points and high assists.
# therefor to find the best players i looked spesficlly at clsuter group 2, the best performing players, and looked for the ones that had a dark color, meaning they were cheap, and had high points and assists.
# i chose to have 3 cluster groups ebucase after perofrming an elbow plot, which is a method to determn whats the right number of clusters to group you uqniue data from to take into account busy and noicy spots.

# after looking at the stats and the graph, i suggst that Mr.Rooney should recruit Russell Westbrook and Isaiah Collier as my top two choicese beucase they had high stats with ____fill in with real number data___.
# this wasy Mr.rooney can save money from teh team to spend on other players, and also have two high performing players that can help the team win games and make the playoffs.
#In terms of who to avoid, I suggest that Mr.Rooney stay away from Bradley Beal and Anthony Davis. They are most very expesntive adn for not good reasons as they underperform bsaed on tehre stats of ___fill in with real num data____.

#to conclude, clsutering helped me identify where to look for the best plaeyrs by givieng me a more detailed and spesific groupgina dn set of payers i can sort and filter through to fine the stars that smiply sorting a long speadsheet might miss.
# it also provided a neat and clean visual that can make it easy to show other stakeholders and deicison makers.
#in the fture this method can also give an easy way to play around with different features additonly to see things like play time and such. 
