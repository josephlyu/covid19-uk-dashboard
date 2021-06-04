'''
Author: Shangjie Lyu
GitHub: https://github.com/josephlyu

The dash app and layout for the index page.
'''

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from layouts import layout_uk, layout_world, layout_map
from figures.figures_uk import (api_uk_last_update, api_uk_date, today_uk_newcases, today_uk_newdeaths,
                                today_uk_cumcases, today_uk_cumdeaths, fig_index_cases, fig_index_deaths)

##### DEFINE GLOBAL VARIABLES #####
SIDEBAR_STYLE = {'position':'fixed', 'top':0, 'left':0, 'bottom':0, 'width':'22rem', 'padding':'1.5rem 1rem'}

CONTENT_STYLE = {'margin-left':'22rem', 'margin-right':'0rem', 'padding':'1rem 1rem', 'background-color':'#e6ecec'}

COLORS = {'side_title':'#ba3a0a', 'side_blue':'#3b6b7b', 'side_red':'#c65d35', 'side_dark':'#666666', 'side_text':'#9e9e9e', 'side_time':'#2196f3'}

##### CREATE APP AND SERVER #####
app = dash.Dash(__name__, title='COVID-19', update_title='Loading...', external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

application = app.server

##### INDEX PAGE LAYOUT #####
sidebar = html.Div([
    html.Img(src='/assets/logo.png', style={'width':'10rem', 'margin-left':'4.83rem'}),
    html.Hr(),

    html.Span('COVID-19', style={'color':COLORS['side_title'], 'font-size':20, 'font-weight':'bold'}),
    html.P('Visualization and Forecasting', style={'color':COLORS['side_dark'], 'font-weight':'bold'}),
    html.Hr(),

    html.P([
        html.Span('Last update: ', style={'color':COLORS['side_text'], 'font-size':13}),
        html.Span(api_uk_last_update, style={'color':COLORS['side_time'], 'font-size':10})
    ]),

    dbc.Row([
        dbc.Col(
            dbc.ButtonGroup([
                dbc.Button('UK', href='/uk', id='uk-link', color='light', style={'width':80, 'color':COLORS['side_blue'], 'font-weight':'bold'}),
                dbc.Button('World', href='/world', id='world-link', color='light', style={'width':80, 'color':COLORS['side_blue'], 'font-weight':'bold'})
            ], size='sm')
        ),
        dbc.Col(
            dbc.Button('Map', href='/map', id='map-link', color='light', size='sm', style={'width':80, 'color':COLORS['side_red'], 'font-weight':'bold', 'float':'right'}))
    ]),

    html.Hr(),
    html.P([
        html.Span('Latest Figures (UK)', style={'color':COLORS['side_text']}),
        html.P(api_uk_date, style={'color':COLORS['side_blue'], 'font-size':25, 'font-weight':'bold'})
    ]),

    dbc.Row([
        dbc.Col(html.Span('New cases', style={'color':COLORS['side_text']})),
        dbc.Col(html.Span('New deaths', style={'color':COLORS['side_text']}))
    ]),
    dbc.Row([
        dbc.Col(html.P(today_uk_newcases, style={'color':COLORS['side_dark'], 'font-size':25, 'font-weight':'bold'})),
        dbc.Col(html.P(today_uk_newdeaths, style={'color':COLORS['side_red'], 'font-size':25, 'font-weight':'bold'}))
    ]),

    dbc.Row([
        dbc.Col(html.Span('Total cases', style={'color':COLORS['side_text']})),
        dbc.Col(html.Span('Total deaths', style={'color':COLORS['side_text']}))
    ]),
    dbc.Row([
        dbc.Col(html.Span(today_uk_cumcases, style={'color':COLORS['side_dark'], 'font-size':25, 'font-weight':'bold'})),
        dbc.Col(html.Span(today_uk_cumdeaths, style={'color':COLORS['side_red'], 'font-size':25, 'font-weight':'bold'}))
    ]),

    html.Hr(),
    html.P('The Curve of Cases', style={'color':COLORS['side_text']}),
    dcc.Graph(figure=fig_index_cases, config={'displayModeBar':False}),

    html.Hr(),
    html.P('The Curve of Deaths', style={'color':COLORS['side_text']}),
    dcc.Graph(figure=fig_index_deaths, config={'displayModeBar':False})
], style=SIDEBAR_STYLE)

content = html.Div(id='page-content', style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id='url'), sidebar, content])

##### INDEX PAGE CALLBACKS #####
@app.callback(
    [Output(f'{s}-link', 'active') for s in ['uk', 'world', 'map']],
    [Input('url', 'pathname')])
def toggle_active_links(pathname):
    if pathname == '/':
        return True, False, False
    return [pathname == f'/{s}' for s in ['uk', 'world', 'map']]

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')])
def render_page_content(pathname):
    if pathname in ['/', '/uk']:
        return layout_uk
    elif pathname == '/world':
        return layout_world
    elif pathname == '/map':
        return layout_map
    return dbc.Jumbotron([
        html.H1('404: Not found', className='text-danger'),
        html.Hr(),
        html.P(f'The pathname {pathname} was not recognised...')
    ])

##### MAIN FUNCTION #####
if __name__ == '__main__':
    app.run_server(debug=True)
