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

# As VP of Analytics for the Wizards, I am helping head scout Mr. Rooney recruit the best 
# players for the best value through data science. The goal is to find the best skill level 
# performers that are underpaid in order to recruit them for cheap and build a playoff ready 
# team without overspending.

# To complete this project I had two initial data sets. One contained information on every NBA 
# player, their team, and their salary. The other listed every NBA player's performance stats. 
# I merged these data sets by player name through an inner merge that only kept players that 
# were in both data sets, so that I had a complete data set with all the information I needed. 
# After merging, I cleaned the data by dropping duplicates after noticing a pattern in the teams 
# column. When a player transferred teams, there was a separate row that listed them as "2TM" 
# and combined their stats into a total. I kept that row and dropped the duplicates. I also 
# dropped columns with mostly missing data, rows with missing salary data, and the percentage 
# columns for the same reason.

# In order to see patterns in this data, I used a method called K-Means Clustering. This method 
# groups players together based on their similarities across chosen features. While you don't 
# necessarily need to know the math behind it, the general idea is that it's calculated on 
# distances and averages between points. This is why it's important to scale the data to one 
# universal scale first, so that the distances aren't calculated on skewed scales that could 
# interfere with how the groups are formed.

# In order to visualize the clustering I had to choose two features to plot on the x and y axis 
# of a scatter plot. I wanted to choose features with a high correlation to salary so that the 
# cluster groups could be distinctly defined, and also a high variance, meaning the data spans 
# a wide range of numbers so we can see real differences between players. I calculated the 
# variance of important features like points scored, minutes played, and assists, then plotted 
# a correlation matrix to see how each feature related to salary. I found that points and assists 
# had the highest correlation with salary and also had high variance, so I chose those two as 
# my x and y axis.

# My target variable, the thing I am trying to draw insights from, is salary. To find the best 
# value players I set the color of the points to represent salary. The darker the color the 
# cheaper the player, and the more teal they get the more expensive they are. The goal is to 
# find the dark colored points in the top right of the graph, which shows players with high 
# performing stats that aren't getting paid a lot. I clustered the data into 3 groups based on 
# an elbow plot, which is a method to determine the right number of clusters for your data. 
# Cluster 0 is the worst performing players with low points and assists. Cluster 1 is the mid 
# tier players with a wide range but generally average stats. Cluster 2 is the best performers 
# with high points and assists. To find the best value players I looked specifically at cluster 2 
# and filtered for the ones with the darkest color, meaning the cheapest salary.

# After looking at the stats and the graph, I suggest that Mr. Rooney recruit Russell Westbrook 
# and Isaiah Collier as my top two choices. Westbrook put up 796 points and 342 assists while 
# only making $2.3M, and Collier had the highest assists in the entire group at 356 with 499 
# points and is only making $2.6M making him a great cheap playmaker. This way Mr. Rooney can 
# save money to spend on other players while also having two high performing players that can 
# help the team win games and make the playoffs. In terms of who to avoid, I suggest Mr. Rooney 
# stay away from Bradley Beal and Anthony Davis. Beal is making $59M and only put up 49 points 
# and 10 assists all season, and Davis is making $54M with only 407 points and 56 assists.

# To conclude, clustering helped me identify where to look for the best players by giving me a 
# more detailed and specific grouping that simply sorting a long spreadsheet might miss. It also 
# produced a clean visual that makes it easy to present to other stakeholders and decision makers. 
# In the future this method could also be used to explore additional features like play time to 
# add even more depth to the analysis.

