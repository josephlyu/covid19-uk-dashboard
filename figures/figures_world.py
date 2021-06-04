'''
Author: Shangjie Lyu
GitHub: https://github.com/josephlyu

The figures for the World page, using data from
Johns Hopkins University's GitHub repository.

Link: https://github.com/CSSEGISandData/COVID-19
'''

import pandas as pd
import plotly.graph_objects as go

##### DEFINE GLOBAL VARIABLES #####
COLORS = {'text':'#114b5f', 'background': '#e6ecec', 'case':'#3f6678', 'death':'#ba3a0a'}
COLORS_MAJOR = {'China':'#cf7d27', 'United States':'#09856c', 'Germany':'#000000', 'France':'#3e82b3',
                'Spain':'#f2ca00', 'Italy':'#800080', 'United Kingdom':'#ba3a0a'}
COLOR_OTHERS = '#bdbdbd'

RENAME = {'Saint Kitts and Nevis':'KNA*', 'Saint Vincent and the Grenadines':'VCT*', 'Antigua and Barbuda':'ATG*',
          'Papua New Guinea':'PNG*', 'Diamond Princess':'Diamond*', 'Sao Tome and Principe':'STP*',
          'Trinidad and Tobago':'TTO*', 'Central African Republic':'CAF*', 'Equatorial Guinea':'GNQ*',
          'Congo (Brazzaville)':'COG*', 'Congo (Kinshasa)':'COD*', 'North Macedonia':'MKD*',
          'Bosnia and Herzegovina':'BIH*', 'West Bank and Gaza':'PSE*', 'United Arab Emirates':'ARE*',
          'Dominican Republic':'DOM*', 'Solomon Islands':'SLB*', 'US':'United States'}

##### LOAD AND PREPROCESS DATA #####
url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
url_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

df_confirmed = pd.read_csv(url_confirmed)
df_deaths = pd.read_csv(url_deaths)

df_confirmed_total = df_confirmed.drop(columns=['Province/State', 'Lat', 'Long']).groupby('Country/Region').sum()
df_deaths_total = df_deaths.drop(columns=['Province/State', 'Lat', 'Long']).groupby('Country/Region').sum()

df_confirmed_total_sorted = df_confirmed_total.sort_values(df_confirmed_total.columns[-1]).T
df_deaths_total_sorted = df_deaths_total.sort_values(df_deaths_total.columns[-1]).T

columns_confirmed = df_confirmed_total_sorted.columns.tolist()
columns_confirmed_new = [RENAME[country] if country in RENAME else country for country in columns_confirmed]
df_confirmed_total_sorted.columns = columns_confirmed_new
for country in ['China', 'United States', 'Germany', 'France', 'Spain', 'Italy', 'United Kingdom']:
    columns_confirmed_new.remove(country)
    columns_confirmed_new.append(country)
df_confirmed_total_sorted = df_confirmed_total_sorted[columns_confirmed_new]

columns_deaths = df_deaths_total_sorted.columns.tolist()
columns_deaths_new = [RENAME[country] if country in RENAME else country for country in columns_deaths]
df_deaths_total_sorted.columns = columns_deaths_new
for country in ['China', 'United States', 'Germany', 'France', 'Spain', 'Italy', 'United Kingdom']:
    columns_deaths_new.remove(country)
    columns_deaths_new.append(country)
df_deaths_total_sorted = df_deaths_total_sorted[columns_deaths_new]

##### FIGURE FOR CUMULATIVE CASES #####
fig_confirmed_cum = go.Figure()
for i in range(185):
    country = df_confirmed_total_sorted.columns[i]
    if df_confirmed_total_sorted[country][-1] < 100: continue
    country_confirmed = df_confirmed_total_sorted[country].loc[df_confirmed_total_sorted[country] >= 100].reset_index(drop=True)
    fig_confirmed_cum.add_scatter(name=country, x=country_confirmed.index, y=country_confirmed.values, line={'color':COLOR_OTHERS, 'width':3})
for i in range(185, 192):
    country = df_confirmed_total_sorted.columns[i]
    country_confirmed = df_confirmed_total_sorted[country].loc[df_confirmed_total_sorted[country] >= 100].reset_index(drop=True)
    fig_confirmed_cum.add_scatter(name=country, x=country_confirmed.index, y=country_confirmed.values, line={'color':COLORS_MAJOR[country], 'width':3})

fig_confirmed_cum.update_xaxes(range=[0, len(df_confirmed_total_sorted)-25], title='Days from 100th recorded case', title_font_size=12)
fig_confirmed_cum.update_yaxes(type='log', dtick=1, showline=True, linewidth=2, linecolor=COLORS['text'], title='Number of cases', title_font_size=12)
fig_confirmed_cum.update_layout(legend={'traceorder':'reversed', 'title':'Double click to<br>select a country', 'font':{'size':10}},
                                font={'color':COLORS['text']}, hovermode='closest', plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['background'],
                                title='Cumulative number of <b>cases</b><br>by number of days since 100th case', title_x=0.07, title_font_size=16)
fig_confirmed_cum.add_annotation(x=0.04, y=1, text='<i>Logarithmic Scale</i>', xref='paper', yref='paper', showarrow=False)

##### FIGURE FOR CUMULATIVE DEATHS #####
fig_deaths_cum = go.Figure()
for i in range(185):
    country = df_deaths_total_sorted.columns[i]
    if df_deaths_total_sorted[country][-1] < 10: continue
    country_deaths = df_deaths_total_sorted[country].loc[df_deaths_total_sorted[country] >= 10].reset_index(drop=True)
    fig_deaths_cum.add_scatter(name=country, x=country_deaths.index, y=country_deaths.values, line={'color':COLOR_OTHERS, 'width':3})
for i in range(185, 192):
    country = df_deaths_total_sorted.columns[i]
    country_deaths = df_deaths_total_sorted[country].loc[df_deaths_total_sorted[country] >= 10].reset_index(drop=True)
    fig_deaths_cum.add_scatter(name=country, x=country_deaths.index, y=country_deaths.values, line={'color':COLORS_MAJOR[country], 'width':3})

fig_deaths_cum.update_xaxes(range=[0, len(df_deaths_total_sorted)-30], title='Days from 10th recorded death', title_font_size=12)
fig_deaths_cum.update_yaxes(type='log', dtick=1, showline=True, linewidth=2, linecolor=COLORS['text'], title='Number of deaths', title_font_size=12)
fig_deaths_cum.update_layout(legend={'traceorder':'reversed', 'title':'Double click to<br>select a country', 'font':{'size':10}},
                             font={'color':COLORS['text']}, hovermode='closest', plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['background'],
                             title='Cumulative number of <b>deaths</b><br>by number of days since 10th death', title_x=0.07, title_font_size=16)
fig_deaths_cum.add_annotation(x=0.04, y=1, text='<i>Logarithmic Scale</i>', xref='paper', yref='paper', showarrow=False)

##### FIGURE FOR CASES TRAJECTORY #####
endpoint_confirmed_traj_x, endpoint_confirmed_traj_y = [], []

fig_confirmed_traj = go.Figure()
for i in range(185):
    country = df_confirmed_total_sorted.columns[i]
    if df_confirmed_total_sorted[country][-1] < 100: continue
    today = df_confirmed_total_sorted[country].rolling(7).mean()
    lastday = df_confirmed_total_sorted[country].shift(7).rolling(7).mean()
    country_confirmed_traj = pd.DataFrame({'cumcases': today, 'newcases': today-lastday})
    fig_confirmed_traj.add_scatter(name=country, x=country_confirmed_traj['cumcases'], y=country_confirmed_traj['newcases'], line={'color':COLOR_OTHERS, 'width':1})
for i in range(185, 192):
    country = df_confirmed_total_sorted.columns[i]
    today = df_confirmed_total_sorted[country].rolling(7).mean()
    lastday = df_confirmed_total_sorted[country].shift(7).rolling(7).mean()
    country_confirmed_traj = pd.DataFrame({'cumcases': today, 'newcases': today-lastday})
    endpoint_confirmed_traj_x.append(country_confirmed_traj['cumcases'][-1])
    endpoint_confirmed_traj_y.append(country_confirmed_traj['newcases'][-1])
    fig_confirmed_traj.add_scatter(name=country, x=country_confirmed_traj['cumcases'], y=country_confirmed_traj['newcases'], line={'color':COLORS_MAJOR[country], 'width':2.5})

fig_confirmed_traj.add_scatter(x=endpoint_confirmed_traj_x, y=endpoint_confirmed_traj_y, mode='markers', marker={'color':COLORS['case'], 'size':4}, showlegend=False)

fig_confirmed_traj.update_xaxes(type='log', range=[2, 7.8], dtick=1, title='Total confirmed cases', title_font_size=12)
fig_confirmed_traj.update_yaxes(type='log', range=[1, 6.7], dtick=1, showline=True, linewidth=2, linecolor=COLORS['text'], title='New cases (in the past week)', title_font_size=12)
fig_confirmed_traj.update_layout(legend={'traceorder':'reversed', 'title':'Double click to<br>select a country', 'font':{'size':10}},
                                 font={'color':COLORS['text']}, hovermode='closest', plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['background'],
                                 title='Trajectory of confirmed <b>cases</b><br><sub>A <i>downward</i> trend indicates a <i>slowing</i> growth rate</sub>', title_x=0.07, title_font_size=16)
fig_confirmed_traj.add_annotation(x=0.04, y=0.98, text='<i>Logarithmic Scale</i>', xref='paper', yref='paper', showarrow=False)

##### FIGURE FOR DEATHS TRAJECTORY #####
endpoint_deaths_traj_x, endpoint_deaths_traj_y = [], []

fig_deaths_traj = go.Figure()
for i in range(185):
    country = df_deaths_total_sorted.columns[i]
    if df_deaths_total_sorted[country][-1] < 10: continue
    today = df_deaths_total_sorted[country].rolling(7).mean()
    lastday = df_deaths_total_sorted[country].shift(7).rolling(7).mean()
    country_deaths_traj = pd.DataFrame({'cumdeaths': today, 'newdeaths': today-lastday})
    fig_deaths_traj.add_scatter(name=country, x=country_deaths_traj['cumdeaths'], y=country_deaths_traj['newdeaths'], line={'color':COLOR_OTHERS, 'width':1})
for i in range(185, 192):
    country = df_deaths_total_sorted.columns[i]
    today = df_deaths_total_sorted[country].rolling(7).mean()
    lastday = df_deaths_total_sorted[country].shift(7).rolling(7).mean()
    country_deaths_traj = pd.DataFrame({'cumdeaths': today, 'newdeaths': today-lastday})
    endpoint_deaths_traj_x.append(country_deaths_traj['cumdeaths'][-1])
    endpoint_deaths_traj_y.append(country_deaths_traj['newdeaths'][-1])
    fig_deaths_traj.add_scatter(name=country, x=country_deaths_traj['cumdeaths'], y=country_deaths_traj['newdeaths'], line={'color':COLORS_MAJOR[country], 'width':2.5})

fig_deaths_traj.add_scatter(x=endpoint_deaths_traj_x, y=endpoint_deaths_traj_y, mode='markers', marker={'color':COLORS['death'], 'size':4}, showlegend=False)

fig_deaths_traj.update_xaxes(type='log', range=[1, 6], dtick=1, title='Total recorded deaths', title_font_size=12)
fig_deaths_traj.update_yaxes(type='log', range=[0, 4.7], dtick=1, showline=True, linewidth=2, linecolor=COLORS['text'], title='New deaths (in the past week)', title_font_size=12)
fig_deaths_traj.update_layout(legend={'traceorder':'reversed', 'title':'Double click to<br>select a country', 'font':{'size':10}},
                              font={'color':COLORS['text']}, hovermode='closest', plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['background'],
                              title='Trajectory of recorded <b>deaths</b><br><sub>A <i>downward</i> trend indicates a <i>slowing</i> growth rate</sub>', title_x=0.07, title_font_size=16)
fig_deaths_traj.add_annotation(x=0.04, y=0.98, text='<i>Logarithmic Scale</i>', xref='paper', yref='paper', showarrow=False)
