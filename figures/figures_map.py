'''
Author: Shangjie Lyu
GitHub: https://github.com/josephlyu

The figures for the Map page, using data from
Public Health Englend's COVID-19 UK API.

Link: https://coronavirus.data.gov.uk/developers-guide
'''

import json
import plotly.express as px

from uk_covid19 import Cov19API

##### DEFINE GLOBAL VARIABLES #####
COLORS = {'text':'#114b5f', 'background': '#e6ecec'}

##### LOAD DATA #####
token = open('data/.mapbox_token').read()

with open('data/boundaries.geojson') as f:
    boundaries = json.load(f)

filters_utla = ['areaType=utla']

structure_utla = {
    'date': 'date',
    'name': 'areaName',
    'code': 'areaCode',
    'cases': 'cumCasesBySpecimenDateRate',
    'deaths': 'cumDeaths28DaysByDeathDateRate'
}

api_utla = Cov19API(filters_utla, structure_utla, latest_by='cumCasesBySpecimenDateRate')
df_utla = api_utla.get_dataframe()

##### MAP FOR CASES #####
fig_cases_map = px.choropleth_mapbox(df_utla, boundaries, featureidkey='properties.code', locations='code', color='cases', hover_name='name',
                                     labels={'code':'Area Code', 'cases':'Cases'}, color_continuous_scale='Blues', zoom=4.9, height=900,
                                     center={'lat':55.8, 'lon':-3.3}, mapbox_style='mapbox://styles/josephlyu/ckmc08wfihnxu17o5a3du4xeq')

fig_cases_map.update_layout(font={'color':COLORS['text']}, paper_bgcolor=COLORS['background'], hoverlabel={'bgcolor':COLORS['background'], 'font_color':COLORS['text']},
                            margin={'l':0, 'r':0, 'b':10, 't':40}, title='Cumulative <b>cases</b> per 100k population', title_x=0.03, mapbox={'accesstoken':token})


fig_cases_map.update_traces(marker_line_width=0.1)

##### MAP FOR DEATHS #####
fig_deaths_map = px.choropleth_mapbox(df_utla, boundaries, featureidkey='properties.code', locations='code', color='deaths', hover_name='name',
                                     labels={'code':'Area Code', 'deaths':'Deaths'}, color_continuous_scale='Reds', zoom=4.9, height=900,
                                     center={'lat':55.8, 'lon':-3.3}, mapbox_style='mapbox://styles/josephlyu/ckmc08wfihnxu17o5a3du4xeq')

fig_deaths_map.update_layout(font={'color':COLORS['text']}, paper_bgcolor=COLORS['background'], hoverlabel={'bgcolor':COLORS['background'], 'font_color':COLORS['text']},
                             margin={'l':0, 'r':0, 'b':10, 't':40}, title='Cumulative <b>deaths</b> per 100k population', title_x=0.03, mapbox={'accesstoken':token})

fig_deaths_map.update_traces(marker_line_width=0.1)
