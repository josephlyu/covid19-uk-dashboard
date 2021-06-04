'''
Author: Shangjie Lyu
GitHub: https://github.com/josephlyu

The layouts for the UK, World and Map page.
'''

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from figures.figures_uk import fig_uk_cases, fig_uk_deaths
from figures.figures_world import (fig_confirmed_cum, fig_deaths_cum, fig_confirmed_traj, fig_deaths_traj)
from figures.figures_map import fig_cases_map, fig_deaths_map

##### CREATE LAYOUTS #####
layout_uk = html.Div([
    dcc.Graph(figure=fig_uk_cases),
    dcc.Graph(figure=fig_uk_deaths)
])

layout_world = html.Div([
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_confirmed_cum)),
        dbc.Col(dcc.Graph(figure=fig_deaths_cum))
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_confirmed_traj)),
        dbc.Col(dcc.Graph(figure=fig_deaths_traj))
    ])
])

layout_map = html.Div([
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_cases_map)),
        dbc.Col(dcc.Graph(figure=fig_deaths_map))
    ])
])
