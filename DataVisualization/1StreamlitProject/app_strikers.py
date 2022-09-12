#from turtle import color, width
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import requests
#from bs4 import BeautifulSoup
#import re
#import time
#import json
#import os
from cProfile import run
import streamlit as st
import seaborn as sns
#import themepy
from highlight_text import HighlightText, ax_text, fig_text
from io import BytesIO
#import shutil 
#import matplotlib as mpl
from mplsoccer.pitch import VerticalPitch
from datetime import datetime
from PIL import Image


#theme = themepy.Theme()
#theme.set_theme("pavel_light")



df=pd.read_csv("/info_epl_strikers.csv")
image_df=pd.read_csv("/images_epl_strikers.csv")
ud_data=pd.read_csv("/understat_epl_strikers.csv")
fb_data=pd.read_csv("/fbref_epl_strikers.csv")

fb_data.columns = fb_data.iloc[0] 
fb_data = fb_data[1:]
fb_data.columns = [*fb_data.columns[:-1], 'name']

fb_uni=fb_data["name"].unique()
ud_uni=ud_data["player"].unique()
commons=np.intersect1d(fb_uni,ud_uni)
ud_data=ud_data[ud_data["player"].isin(commons)]
fb_data=fb_data[fb_data["name"].isin(commons)]
image_df=image_df[image_df["Name"].isin(commons)]
df=df[df["Name"].isin(commons)]
ud_data=ud_data[ud_data["season"]==2021]
image_df.reset_index(drop=True,inplace=True)
df.reset_index(drop=True,inplace=True)
ud_data.reset_index(drop=True,inplace=True)
fb_data.reset_index(drop=True,inplace=True)

fb_data=fb_data[fb_data["Comp"]=="Premier League"]
new = fb_data["Round"].str.split(" ", n = 1, expand = True)
fb_data["Gameweek"]= new[1]
fb_data["Gameweek"]=fb_data["Gameweek"].astype(int)
fb_data.drop(columns =["Round"], inplace = True)
fb_data["Date"] = pd.to_datetime(fb_data["Date"])

radar_df=fb_data




img=Image.open('Images\\ezz.jpg')
st.set_page_config(layout="wide", page_title='Finishing Dashboard',page_icon=img)

col1,col2=st.columns([3,1])

col1.markdown("# 2020/2021 EPL Finishing Dashboard")

col2.markdown("Data: **fbref.com** and **understat.com**  \n Made by: **@pawellkaroll**")


col1,col2,col3=st.columns([1,1,2])

names=list(df["Name"].drop_duplicates())
name_choice=col1.selectbox("Select a Player:",names)
gw_choice=col3.slider("Select a Range of Gameweeks:",min_value=1, max_value=38,value=[1,38],step=1)


df=df[df["Name"]==name_choice]
df=df[["Name", "Club","Nation","Position","Age"]]
fb_data=fb_data[(fb_data["Gameweek"]>=gw_choice[0])&(fb_data["Gameweek"]<=gw_choice[1])]
fb_data=fb_data[fb_data["name"]==name_choice]
fb_data=fb_data.sort_values("Gameweek")






col1,col2,col3=st.columns([1,1,2])


col1.markdown("### Image")
temp=image_df[image_df["Name"]==name_choice]
image_url = str(temp.iloc[0, 7])
col1.image(image_url)

col2.markdown("### Info")

mga_data=fb_data[['Gls','Ast']]
mga_data=mga_data.replace("On matchday squad, but did not play","0")
mga_data.reset_index(drop=True,inplace=True)
mga_data['Matches']=1
mga_data=mga_data.astype(float)
mga_data=mga_data.astype(int)
mga_data=mga_data.sum()

col2.markdown("Name: **{}**  \n Club: **{}**  \n Nationality: **{}**  \n Position: **{}**  \n Age: **{}**  \n Matches/Goals/Assists: **{}/{}/{}**"
.format(df.iloc[0,0],df.iloc[0,1],df.iloc[0,2],df.iloc[0,3],df.iloc[0,4],mga_data[2],mga_data[0],mga_data[1]))
#col2.table(df)

col3.markdown("### Form")

form_df=fb_data.iloc[-5:]
if form_df.empty:
    col3.markdown("Not enough data")
else:
    form_df=form_df[['Date','Opponent','Min','Gls','Ast','Sh','Cmp']]
    form_df=form_df.replace("On matchday squad, but did not play","0")
    form_df['Date'] = form_df['Date'].apply(lambda x: x.strftime("%m/%d/%Y"))
    form_df["Match"]=form_df["Date"]+" vs "+form_df["Opponent"]
    form_df=form_df.set_index("Match")
    form_df=form_df[['Min','Gls','Ast','Sh','Cmp']]
    form_df=form_df.astype(float)
    form_df=form_df.astype(int)
    cm = sns.light_palette("green", as_cmap=True)
    form_df=form_df.style.background_gradient(cmap=cm)
    col3.table(form_df)




col1,col2,col3=st.columns([3,2,1])

col1.markdown("### Shoting Ability  \n **4-game Rolling Average**")

#DATA######################

goals_df=fb_data[["name","Gls","xG","Gameweek"]]
goals_df=goals_df[goals_df["name"]==name_choice]
goals_df=goals_df.dropna()
goals_df = goals_df[goals_df["xG"].str.contains("On matchday squad, but did not play") == False]
goals_df["xG"]=pd.to_numeric(goals_df["xG"])
goals_df["Gls"]=pd.to_numeric(goals_df["Gls"])
goals_df.reset_index(drop=True,inplace=True)
goals_df["diff"]=goals_df["Gls"]-goals_df["xG"]
goals_df["diff_MA5"]=goals_df["diff"].rolling(5).mean()
goals_df["sum"]=goals_df["diff_MA5"].cumsum()

#PLOTTING####################################
if len(goals_df.index)<6:
    col1.markdown("Not enough data")
else:
    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(600*px, 350*px))
    ax.axis('off')

    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9)
    grid = plt.GridSpec(1, 1,wspace=0, hspace=0)
    ax=fig.add_subplot(grid[:, :])

    ax.set_facecolor('#F5F5DC')
    ax.xaxis.get_major_locator().set_params(integer=True)

    x=goals_df.index.values
    y=goals_df["sum"]
    ax.plot(x, y,c="#50C878",alpha=0.6,linewidth=3,zorder=4)
    ax.axhline(y=0, color='black',zorder=1,ls="--",lw=2,alpha=0.6)

    ax.set_xlabel("Games", fontsize=10,fontweight='medium')
    ax.set_ylabel("Goals-xG", fontsize=10,fontweight='medium')
    ax.tick_params(labelsize=8)
    
    ax.grid()

    buf = BytesIO()
    fig.savefig(buf, format="png")
    col1.image(buf)


col2.markdown("### Radar Chart  \n **Percentiles/Stats per 90**")

#DATA######################

radar_df=radar_df[(radar_df["Gameweek"]>=gw_choice[0])&(radar_df["Gameweek"]<=gw_choice[1])]
radar_df=radar_df[["Min","xG","xA","Sh","Press","Tkl","Int","SCA", "Carries","name"]]
radar_df=radar_df.dropna()
radar_df = radar_df[radar_df["xG"].str.contains("On matchday squad, but did not play") == False]
radar_df[["Min","xG","xA","Sh","Press","Tkl","Int","SCA","Carries"]]=radar_df[["Min","xG","xA","Sh","Press","Tkl","Int","SCA","Carries"]].apply(lambda x:pd.to_numeric(x))
radar_df.reset_index(drop=True,inplace=True)
radar_df=radar_df.groupby("name").sum()
radar_df[["xG","xA","Sh","Press","Tkl","Int","SCA","Carries"]]=radar_df[["xG","xA","Sh","Press","Tkl","Int","SCA","Carries"]].apply(lambda x:x/(radar_df["Min"]/90))
radar_df.reset_index(inplace=True)
labels=list(radar_df.columns)
labels.remove("Min")

#PLOTTING####################################

def make_radar(df,labels, player1):
    
    df_resized=df[labels]


    #PERCENTILES
    labels_per90=list(df_resized.columns)
    labels_per90.remove("name")
    labels_per90=list(labels_per90)
    labels_per90=np.array(labels_per90)
    df= df_resized.loc[:, df_resized.columns != 'name'].rank(pct=True).round(decimals=2)
    df=pd.concat([df_resized["name"],df],axis=1)

    #DEFINING DF
    p1=df[df["name"]==player1]
    p1=p1.reset_index()


    #MAKING RADAR
    N = len(labels_per90)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(350*px,350*px))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
 

    plt.xticks(angles[:-1], labels_per90, color='black', size=10, fontname='Open Sans', fontweight='medium')
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction('clockwise')
 
    ax.set_rlabel_position(0)
    plt.yticks([0.2,0.4,0.6,0.8,1.0], ['20th','40th','60th','80th','100th'], color="black", 
               size=8, fontname='Open Sans', fontweight='light' )
    plt.ylim(0,1)
 
    # player
    values=p1.loc[0,labels_per90].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label="Player 1", color='#50C878')
    ax.fill(angles, values, '#50C878', alpha=0.25)
    
    
    ax.grid(color='#ECECEE')
    ax.spines['polar'].set_color("w")
    ax.set_facecolor("#F5F5DC")
    
    buf = BytesIO()
    fig.savefig(buf, format="png")
    return buf

temp=radar_df[radar_df["name"]==name_choice]

if temp.empty:
    col2.markdown("Not enough data")
else:
    buf=make_radar(df=radar_df,labels=labels,player1=name_choice)
    col2.image(buf)

col3.text("")
col3.text("")
col3.text("")
col3.text("")
col3.text("")
col3.text("")
col3.text("")
col3.text("")
col3.text("")
col3.text("")
col3.markdown('xG - expected goals  \n xA - expected assists  \n Sh - shots attempted  \n Press - pressures  \n Tkl - tackles  \n Int - interceptions  \n SCA - shot creating actions  \n Carries - carries via dribble or pass')



col1,col2=st.columns([1,1])

col1.markdown("### xG Shot Map")

#DATA######################

ud_data['date']=ud_data['date'].apply(lambda x: x.split(' ',1)[0])
ud_data['Date']=ud_data['date']
ud_data['name']=ud_data['player']
ud_data['Date']=pd.to_datetime(ud_data['date'])
tmp=fb_data[['name','Gameweek','Date']]
temp=pd.merge(tmp,ud_data, on=['Date','name'], how='inner')


shot_df=temp[['minute','result','player','date','X','Y','xG']]
shot_df[['X']]=shot_df[['X']].apply(lambda x: (x/100)*120*100)
shot_df[['Y']]=shot_df[['Y']].apply(lambda x: (x/100)*80*100)
shot_df=shot_df[shot_df['player']==name_choice]

hist_df=temp[['shotType']]




#PLOTTING####################################
if shot_df.empty:
    col1.markdown("Not enough data")
else:
    px = 1/plt.rcParams['figure.dpi']

    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#F5F5DC', line_color='black',half=True)
    fig, ax = pitch.draw(figsize=(500*px, 350*px), constrained_layout=True, tight_layout=False)
    fig.set_facecolor('w')
    plt.gca().invert_xaxis()
    
    shot_df_2=shot_df[shot_df['result']!='Goal']
    shot_df=shot_df[shot_df['result']=='Goal']

    sc2=pitch.scatter(shot_df_2['X'],shot_df_2['Y'],ax=ax,c='None',edgecolors="black",s=shot_df_2['xG']*500,hatch='///',label='No goal')
    sc1=pitch.scatter(shot_df['X'],shot_df['Y'],ax=ax,color='#50C878',edgecolors="black",s=shot_df['xG']*500,label='Goal')
    
    legend = ax.legend(loc="upper center",bbox_to_anchor= (0.14, 0.88),labelspacing=1.3,prop={'weight':'medium','size':10},frameon=False)
    legend.legendHandles[0]._sizes = [400]
    legend.legendHandles[1]._sizes = [400]

    mSize = [0.05,0.10,0.2,0.4,0.6,0.8]
    mSizeS = [500 * i for i in mSize]
    mx = [6,6,6,6,6,6]
    my = [100,103,106,109,112,115]
    plt.scatter(mx,my,s=mSizeS,facecolors="#50C878", edgecolor="black")
    for i in range(len(mx)):
        plt.text(mx[i]+ 5, my[i], mSize[i], fontsize=10, color="black",ha="center", va="center",fontweight='medium')

    buf = BytesIO()
    fig.savefig(buf, format="png")
    col1.image(buf)


#HISTOGRAM
col2.markdown('### Shot Type')

px = 1/plt.rcParams['figure.dpi']
fig, ax = plt.subplots(figsize=(600*px, 350*px))
ax.axis('off')

plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9)
grid = plt.GridSpec(1, 1,wspace=0, hspace=0)
ax=fig.add_subplot(grid[:, :])

ax.set_facecolor('#F5F5DC')

ax.hist(hist_df,color="#50C878",alpha=0.6,linewidth=3,zorder=4)

ax.set_xlabel("Shot Type", fontsize=10,fontweight='medium')
ax.set_ylabel("Count", fontsize=10,fontweight='medium')
ax.tick_params(labelsize=8)

ax.grid()

buf = BytesIO()
fig.savefig(buf, format="png")
col2.image(buf)
