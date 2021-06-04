'''
Author: Shangjie Lyu
GitHub: https://github.com/josephlyu

The figures for the UK page, using data from
Public Health Englend's COVID-19 UK API and
Oxford University's GitHub repository.

Link1: https://coronavirus.data.gov.uk/developers-guide
Link2: https://github.com/OxCGRT/covid-policy-tracker
'''

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from uk_covid19 import Cov19API
import datetime as dt

from pmdarima.arima import auto_arima
from tensorflow.keras import Model, Input, callbacks
from tensorflow.keras.layers import LSTM, Dense, Dropout

##### DEFINE GLOBAL VARIABLES #####
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

COLORS = {'text':'#114b5f', 'background': '#e6ecec', 'case':'#3f6678', 'death':'#ba3a0a', 'pred':'#d7d9db',
          'case_scatter':'#7498a8', 'case_pred':'#005f86', 'death_scatter':'#d88f74', 'death_pred':'#b8272e',
          'index_case':'#666666', 'index_death':'#c65d35', 'index_text':'#9e9e9e'}

FIXED_ANNOTATIONS = [dict(x=-0.03, y=1.15, text='<b><i>Model Selection:</i></b>', font={'size':13}, xref='paper', yref='paper', showarrow=False),
                     dict(x=0.875, y=1.17, text='<b><i>Major<br>Events</i></b>', xref='paper', yref='paper', showarrow=False)]

UK_CASES_EVENTS = [dict(x='2020-3-23', y=1198, text='National Lockdown', ax=-30, ay=-40, arrowhead=5, arrowcolor=COLORS['text'], arrowwidth=1),
                   dict(x='2020-7-4', y=575, text='(Eat out to Help out)<br>Lockdown Eased', ax=-20, ay=-35, arrowhead=5, arrowcolor=COLORS['text'], arrowwidth=1),
                   dict(x='2020-9-7', y=2532, text='University Students Return', ax=-65, ay=-65, arrowhead=5, arrowcolor=COLORS['text'], arrowwidth=1),
                   dict(x='2020-11-5', y=22826, text='Second Lockdown', ax=-90, ay=-25, arrowhead=5, arrowcolor=COLORS['text'], arrowwidth=1),
                   dict(x='2020-12-2', y=14400, text='(Christmas Period)<br>Lockdown Eased', ax=-45, ay=-90, arrowhead=5, arrowcolor=COLORS['text'], arrowwidth=1),
                   dict(x='2020-12-21', y=34396, text='Mass Vaccination<br>(1% Population)', ax=-45, ay=-65, arrowhead=5, arrowcolor=COLORS['text'], arrowwidth=1),
                   dict(x='2021-1-5', y=59344, text='Third Lockdown', ax=-70, ay=-20, arrowhead=5, arrowcolor=COLORS['text'], arrowwidth=1),
                   dict(x='2021-1-12', y=51221, text='Mass Vaccination<br>(5% Population)', ax=80, ay=-15, arrowhead=5, arrowcolor=COLORS['text'], arrowwidth=1),
                   dict(x='2021-2-5', y=17714, text='Mass Vaccination<br>(20% Population)', ax=55, ay=-55, arrowhead=5, arrowcolor=COLORS['text'], arrowwidth=1),
                   dict(x='2021-3-19', y=5485, text='Mass Vaccination<br>(50% Population)', ax=15, ay=-40, arrowhead=5, arrowcolor=COLORS['text'], arrowwidth=1)]

UK_DEATHS_EVENTS = [dict(x='2020-3-23', y=103, text='National Lockdown', ax=-55, ay=-35, arrowhead=5, arrowcolor=COLORS['text'], arrowwidth=1),
                    dict(x='2020-7-4', y=43, text='(Eat out to Help out)<br>Lockdown Eased', ax=15, ay=-55, arrowhead=5, arrowcolor=COLORS['text'], arrowwidth=1),
                    dict(x='2020-9-7', y=12, text='University Students Return', ax=-25, ay=-25, arrowhead=5, arrowcolor=COLORS['text'], arrowwidth=1),
                    dict(x='2020-11-5', y=332, text='Second Lockdown', ax=-85, ay=-10, arrowhead=5, arrowcolor=COLORS['text'], arrowwidth=1),
                    dict(x='2020-12-2', y=427, text='(Christmas Period)<br>Lockdown Eased', ax=-100, ay=-35, arrowhead=5, arrowcolor=COLORS['text'], arrowwidth=1),
                    dict(x='2020-12-21', y=512, text='Mass Vaccination<br>(1% Population)', ax=-80, ay=-65, arrowhead=5, arrowcolor=COLORS['text'], arrowwidth=1),
                    dict(x='2021-1-5', y=809, text='Third Lockdown', ax=-70, ay=-65, arrowhead=5, arrowcolor=COLORS['text'], arrowwidth=1),
                    dict(x='2021-1-12', y=1066, text='Mass Vaccination<br>(5% Population)', ax=-50, ay=-75, arrowhead=5, arrowcolor=COLORS['text'], arrowwidth=1),
                    dict(x='2021-2-5', y=891, text='Mass Vaccination<br>(20% Population)', ax=65, ay=-50, arrowhead=5, arrowcolor=COLORS['text'], arrowwidth=1),
                    dict(x='2021-3-19', y=85, text='Mass Vaccination<br>(50% Population)', ax=30, ay=-50, arrowhead=5, arrowcolor=COLORS['text'], arrowwidth=1)]

CASES_UPDATE_MENUS = [dict(type='dropdown', direction='down', x=0.214, y=1.17, buttons=list([
                        dict(label='Ensemble', method='update', args=[{'visible': [True, True, True, True, True, False, False, False, False, False, False]}]),
                        dict(label='ARIMA', method='update', args=[{'visible': [True, True, False, False, False, True, True, True, False, False, False]}]),
                        dict(label='LSTM', method='update', args=[{'visible': [True, True, False, False, False, False, False, False, True, True, True]}])])),
                      dict(type='buttons', direction='right', x=1, y=1.17, buttons=list([
                        dict(label='Show', method='update', args=[{}, {'annotations': UK_CASES_EVENTS + FIXED_ANNOTATIONS}]),
                        dict(label='Hide', method='update', args=[{}, {'annotations': FIXED_ANNOTATIONS}])]))]

DEATHS_UPDATE_MENUS = [dict(type='dropdown', direction='down', x=0.214, y=1.17, buttons=list([
                         dict(label='Ensemble', method='update', args=[{'visible': [True, True, True, True, True, False, False, False, False, False, False]}]),
                         dict(label='ARIMA', method='update', args=[{'visible': [True, True, False, False, False, True, True, True, False, False, False]}]),
                         dict(label='LSTM', method='update', args=[{'visible': [True, True, False, False, False, False, False, False, True, True, True]}])])),
                       dict(type='buttons', direction='right', x=1, y=1.17, buttons=list([
                         dict(label='Show', method='update', args=[{}, {'annotations': UK_DEATHS_EVENTS + FIXED_ANNOTATIONS}]),
                         dict(label='Hide', method='update', args=[{}, {'annotations': FIXED_ANNOTATIONS}])]))]

##### DEFINE FUNCTIONS TO CONSTRUCT MODELS #####
def to_feature(feature_series, cases=True):
    target_index = INDEX_CASES if cases else INDEX_DEATHS
    feature_index = feature_series.index

    if target_index[0] in feature_index:
        feature_index = feature_series[target_index[0]:].index
        feature_series = feature_series[feature_index]
        if target_index[-1] in feature_index:
            return feature_series[:target_index[-1]].tolist()
        else:
            padding_right = [feature_series[-1] for n in range(len(target_index)-len(feature_index))]
            return feature_series.tolist() + padding_right

    else:
        if target_index[-1] in feature_index:
            feature_index  = feature_series[:target_index[-1]].index
            feature_series = feature_series[feature_index]
            padding_left = [feature_series[0] for n in range(len(target_index)-len(feature_index))]
            return padding_left + feature_series.tolist()
        else:
            padding_left = [feature_series[0] for n in range(target_index.tolist().index(feature_index[0]))]
            padding_right = [feature_series[-1] for n in range(len(target_index)-len(feature_index)-len(padding_left))]
            return padding_left + feature_series.tolist() + padding_right

def to_sequence(data, features=[], input_size=7, output_size=21):
    x, y, arrs = [], [], [data] + features
    for i in range(len(data)-input_size-output_size+1):
        x.append([[arr[n] for arr in arrs] for n in range(i,i+input_size)])
        y.append(data[i+input_size:i+input_size+output_size])
    return np.array(x), np.array(y)

def scale(original):
    return [(n-min(original)) / (max(original)-min(original)) for n in original]

def unscale(scaled, cases=True):
    original = DATA_UK_CASES_DIFF_LIST if cases else DATA_UK_DEATHS_DIFF_LIST
    return [n*(max(original)-min(original)) + min(original) for n in scaled]

def undifference(difference, start_index, cases=True):
    start = DATA_UK_CASES_LIST[start_index] if cases else DATA_UK_DEATHS_LIST[start_index]
    undifferenced = [difference[0] + start]
    for i in range(1, len(difference)):
        undifferenced.append(difference[i] + undifferenced[i-1])
    return undifferenced

def get_original(difference_scaled, start_index, cases=True):
    return undifference(unscale(difference_scaled, cases), start_index, cases)

def predict_LSTM(model, latest_data, cases=True):
    return get_original(model(latest_data).numpy()[0], -1, cases)

def result_LSTM(model, latest_data, cases=True, output_size=21, n_iter=10):
    outputs = np.zeros((n_iter, output_size))
    for i in range(n_iter):
        outputs[i] = predict_LSTM(model, latest_data, cases)
    return outputs.mean(axis=0), np.percentile(outputs,2.5,axis=0), np.percentile(outputs,97.5,axis=0)

def construct_LSTM(cases=True):
    num_features = 4 if cases else 3
    inputs = Input(shape=(None, num_features))
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.2)(x, training=True)
    x = LSTM(64)(x)
    x = Dropout(0.2)(x, training=True)
    x = Dense(128, 'relu')(x)
    x = Dense(64, 'relu')(x)
    outputs = Dense(21)(x)
    model = Model(inputs, outputs)

    if cases:
        x_train, y_train = to_sequence(DATA_UK_CASES_DIFF_LIST_SCALED, [FEATURE_STRINGENCY_FOR_CASES, FEATURE_VACCINATION_FOR_CASES, FEATURE_TESTS_FOR_CASES])
        latest_data, _ = to_sequence(DATA_UK_CASES_DIFF_LIST_SCALED[-7:], [FEATURE_STRINGENCY_FOR_CASES[-7:], FEATURE_VACCINATION_FOR_CASES[-7:], FEATURE_TESTS_FOR_CASES[-7:]], output_size=0)
    else:
        x_train, y_train = to_sequence(DATA_UK_DEATHS_DIFF_LIST_SCALED, [FEATURE_CASES_FOR_DEATHS, FEATURE_VACCINATION_FOR_DEATHS])
        latest_data, _ = to_sequence(DATA_UK_DEATHS_DIFF_LIST_SCALED[-7:], [FEATURE_CASES_FOR_DEATHS[-7:], FEATURE_VACCINATION_FOR_DEATHS[-7:]], output_size=0)

    earlyStopping = callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    model.compile('adam', 'mse')
    model.fit(x_train, y_train, callbacks=earlyStopping, epochs=300, verbose=0)

    return result_LSTM(model, latest_data, cases)

def construct_ARIMA(cases=True):
    data = DATA_UK_CASES_LIST if cases else DATA_UK_DEATHS_LIST

    model = auto_arima(data, seasonal=False, test='adf', information_criterion='bic',
                       error_action='ignore', suppress_warnings=True, njob=-1)

    return model.predict(21, return_conf_int=True)

##### LOAD DATA #####
filters_uk = ['areaType=overview']

structure_total = {
    'date': 'date',
    'newcases': 'newCasesByPublishDate',
    'cumcases': 'cumCasesByPublishDate',
    'newdeaths': 'newDeaths28DaysByPublishDate',
    'cumdeaths': 'cumDeaths28DaysByPublishDate'
}

api_uk = Cov19API(filters_uk, structure_total)

df_uk = api_uk.get_dataframe()
df_uk['date'] = df_uk['date'].astype('datetime64[ns]')

df_uk_cases = df_uk.query('cumcases >= 1')
df_uk_deaths = df_uk.query('cumdeaths >= 1')

##### COMPONENTS FOR INDEX PAGE #####
api_uk_timestamp = api_uk.last_update
api_uk_last_update = api_uk_timestamp[:10] + ' ' + api_uk_timestamp[11:19] + ' UTC'
api_uk_date_str = dt.datetime.strptime(api_uk_timestamp[2:10], '%y-%m-%d')
api_uk_date = api_uk_date_str.strftime('%d %B, %Y')
today_uk_newcases, today_uk_newdeaths = df_uk['newcases'][0], df_uk['newdeaths'][0]
today_uk_cumcases, today_uk_cumdeaths = df_uk['cumcases'][0], df_uk['cumdeaths'][0]

fig_index_cases = go.Figure()
fig_index_cases.add_scatter(x=df_uk_cases['date'], y=df_uk_cases['cumcases'], line={'color':COLORS['index_case']}, fill='tozeroy')
fig_index_cases.update_layout(font={'color':COLORS['index_text']}, hovermode='closest', template='none', margin={'l':0, 'r':0, 't':10, 'b':25}, height=130)
fig_index_cases.update_xaxes(showgrid=False, showline=True, linecolor=COLORS['index_text'], tickformat='%d/%m')
fig_index_cases.update_yaxes(nticks=3)

fig_index_deaths = go.Figure()
fig_index_deaths.add_scatter(x=df_uk_cases['date'], y=df_uk_cases['cumdeaths'], line={'color':COLORS['index_death']}, fill='tozeroy')
fig_index_deaths.update_layout(font={'color':COLORS['index_text']}, hovermode='closest', template='none', margin={'l':0, 'r':0, 't':10, 'b':25}, height=130)
fig_index_deaths.update_xaxes(showgrid=False, showline=True, linecolor=COLORS['index_text'], tickformat='%d/%m')
fig_index_deaths.update_yaxes(nticks=3)

##### PREPARE DATA FOR FORECASTING #####
data_uk_cases = df_uk_cases[['date', 'newcases']].sort_index(ascending=False).set_index('date')
data_uk_cases_avg = data_uk_cases['newcases'].rolling(7).mean().round()[6:]
data_uk_cases_avg_diff = data_uk_cases_avg.diff()[1:]

LEN_CASES =  len(data_uk_cases_avg_diff)
INDEX_CASES = data_uk_cases_avg_diff.index
DATA_UK_CASES_LIST = data_uk_cases_avg.tolist()
DATA_UK_CASES_DIFF_LIST = data_uk_cases_avg_diff.tolist()
DATA_UK_CASES_DIFF_LIST_SCALED = scale(DATA_UK_CASES_DIFF_LIST)

data_uk_deaths = df_uk_deaths[['date', 'newdeaths']].sort_index(ascending=False).set_index('date')
data_uk_deaths_avg = data_uk_deaths['newdeaths'].rolling(7).mean().round()[6:]
data_uk_deaths_avg_diff = data_uk_deaths_avg.diff()[1:]

LEN_DEATHS =  len(data_uk_deaths_avg_diff)
INDEX_DEATHS = data_uk_deaths_avg_diff.index
DATA_UK_DEATHS_LIST = data_uk_deaths_avg.tolist()
DATA_UK_DEATHS_DIFF_LIST = data_uk_deaths_avg_diff.tolist()
DATA_UK_DEATHS_DIFF_LIST_SCALED = scale(DATA_UK_DEATHS_DIFF_LIST)

FEATURE_CASES_FOR_DEATHS = DATA_UK_CASES_DIFF_LIST_SCALED[-LEN_DEATHS:]
FEATURE_DEATHS_FOR_CASES = [0 for n in range(LEN_CASES-LEN_DEATHS)] + DATA_UK_DEATHS_DIFF_LIST_SCALED

url_stringency = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/timeseries/stringency_index.csv'
df_stringency = pd.read_csv(url_stringency, index_col=0)
df_stringency_uk = df_stringency.query('country_code == "GBR"').T[2:]
df_stringency_uk.index = pd.to_datetime(df_stringency_uk.index)
df_stringency_uk.columns = ['stringency']
df_stringency_uk = df_stringency_uk.fillna(method='pad')
stringency_uk = df_stringency_uk['stringency']

stringency_uk_cases, stringency_uk_deaths = to_feature(stringency_uk), to_feature(stringency_uk, False)
FEATURE_STRINGENCY_FOR_CASES, FEATURE_STRINGENCY_FOR_DEATHS = scale(stringency_uk_cases), scale(stringency_uk_deaths)

url_vaccination_uk = 'https://api.coronavirus.data.gov.uk/v2/data?areaType=overview&metric=cumVaccinationFirstDoseUptakeByPublishDatePercentage&format=csv'
df_vaccination_uk = pd.read_csv(url_vaccination_uk, index_col=3)
df_vaccination_uk.index = pd.to_datetime(df_vaccination_uk.index)
vaccination_uk = df_vaccination_uk['cumVaccinationFirstDoseUptakeByPublishDatePercentage'].copy()
vaccination_uk_padding_index = pd.date_range(end=vaccination_uk.index[-1], periods=35)
vaccination_uk_padding = np.linspace(0, vaccination_uk[-1], len(vaccination_uk_padding_index))
for i in range(1, len(vaccination_uk_padding_index)+1):
    vaccination_uk[vaccination_uk_padding_index[-i]] = vaccination_uk_padding[-i]
vaccination_uk = vaccination_uk[::-1]

vaccination_uk_cases, vaccination_uk_deaths = to_feature(vaccination_uk), to_feature(vaccination_uk, False)
FEATURE_VACCINATION_FOR_CASES, FEATURE_VACCINATION_FOR_DEATHS = scale(vaccination_uk_cases), scale(vaccination_uk_deaths)

url_tests = 'https://api.coronavirus.data.gov.uk/v2/data?areaType=overview&metric=newTestsByPublishDate&format=csv'
df_tests_uk = pd.read_csv(url_tests, index_col=3)
df_tests_uk.index = pd.to_datetime(df_tests_uk.index)
tests_uk = df_tests_uk['newTestsByPublishDate'].copy()
tests_padding_index = pd.date_range(end=tests_uk.index[-1], periods=23)
tests_padding = np.linspace(0, tests_uk[-1], len(tests_padding_index))
for i in range(1, len(tests_padding_index)+1):
    tests_uk[tests_padding_index[-i]] = tests_padding[-i]
tests_uk = tests_uk[::-1]

tests_uk_cases = to_feature(tests_uk)
FEATURE_TESTS_FOR_CASES = scale(tests_uk_cases)

##### MODELS FOR CASES FORECASTING #####
pred_uk_cases_ARIMA, conf_uk_cases_ARIMA = construct_ARIMA()
pred_uk_cases_LSTM, min_uk_cases_LSTM, max_uk_cases_LSTM = construct_LSTM()

index_pred_uk_cases = pd.date_range(data_uk_cases.index[-1], periods=22, freq='D', closed='right')
df_pred_uk_cases = pd.DataFrame({'pred_ARIMA': pred_uk_cases_ARIMA, 'min_ARIMA': conf_uk_cases_ARIMA[:,0], 'max_ARIMA': conf_uk_cases_ARIMA[:,1]})

df_pred_uk_cases['pred_LSTM'] = pred_uk_cases_LSTM
df_pred_uk_cases['min_LSTM'] = min_uk_cases_LSTM
df_pred_uk_cases['max_LSTM'] = max_uk_cases_LSTM

df_pred_uk_cases['pred_ENSEMBLE'] = df_pred_uk_cases[['pred_ARIMA', 'pred_LSTM']].mean(axis=1)
df_pred_uk_cases['min_ENSEMBLE'] = df_pred_uk_cases[['min_ARIMA', 'min_LSTM']].mean(axis=1)
df_pred_uk_cases['max_ENSEMBLE'] = df_pred_uk_cases[['max_ARIMA', 'max_LSTM']].mean(axis=1)

##### FIGURE FOR CASES FORECASTING #####
df_pred_uk_cases = df_pred_uk_cases.clip(lower=0)
average_uk_cases = df_uk_cases['newcases'].rolling(7,center=True).mean().round()

fig_uk_cases = go.Figure()
fig_uk_cases.add_bar(name='Recorded', x=df_uk_cases['date'], y=df_uk_cases['newcases'], marker={'color':COLORS['case_scatter']})
fig_uk_cases.add_scatter(name='7-Day Average (Recorded)', x=df_uk_cases['date'], y=average_uk_cases, line={'color':COLORS['case'], 'width':3})
fig_uk_cases.add_scatter(name='Max Predicted', x=index_pred_uk_cases, y=df_pred_uk_cases['max_ENSEMBLE'], line={'color':COLORS['pred']}, showlegend=False)
fig_uk_cases.add_scatter(name='Min Predicted', x=index_pred_uk_cases, y=df_pred_uk_cases['min_ENSEMBLE'], line={'color':COLORS['pred'], 'width':1}, fill='tonexty', showlegend=False)
fig_uk_cases.add_scatter(name='Predicted', x=index_pred_uk_cases, y=df_pred_uk_cases['pred_ENSEMBLE'], line={'color':COLORS['case_pred'], 'width':3})
fig_uk_cases.add_scatter(name='Max Predicted', x=index_pred_uk_cases, y=df_pred_uk_cases['max_ARIMA'], line={'color':COLORS['pred']}, showlegend=False, visible=False)
fig_uk_cases.add_scatter(name='Min Predicted', x=index_pred_uk_cases, y=df_pred_uk_cases['min_ARIMA'], line={'color':COLORS['pred'], 'width':1}, fill='tonexty', showlegend=False, visible=False)
fig_uk_cases.add_scatter(name='Predicted', x=index_pred_uk_cases, y=df_pred_uk_cases['pred_ARIMA'], line={'color':COLORS['case_pred'], 'width':3}, visible=False)
fig_uk_cases.add_scatter(name='Max Predicted', x=index_pred_uk_cases, y=df_pred_uk_cases['max_LSTM'], line={'color':COLORS['pred']}, showlegend=False, visible=False)
fig_uk_cases.add_scatter(name='Min Predicted', x=index_pred_uk_cases, y=df_pred_uk_cases['min_LSTM'], line={'color':COLORS['pred'], 'width':1}, fill='tonexty', showlegend=False, visible=False)
fig_uk_cases.add_scatter(name='Predicted', x=index_pred_uk_cases, y=df_pred_uk_cases['pred_LSTM'], line={'color':COLORS['case_pred'], 'width':3}, visible=False)
fig_uk_cases.update_layout(updatemenus=CASES_UPDATE_MENUS, annotations=UK_CASES_EVENTS + FIXED_ANNOTATIONS, hovermode='x unified', font={'color':COLORS['text']},
                           plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['background'], legend={'orientation':'h', 'traceorder':'normal', 'xanchor':'center', 'x':0.5},
                           title='<b>Covid-19 Cases Forecasting for the UK</b><br><sub>The forecast contains <i>three-week</i> prediction with <i>95%</i> confidence interval</sub>', title_x=0.5)
fig_uk_cases.update_xaxes(range=[str(df_uk_cases['date'].iloc[-1].date()), str(index_pred_uk_cases[-1].date())],
                          showline=True, linecolor=COLORS['text'], mirror=True)
fig_uk_cases.update_yaxes(showline=True, linecolor=COLORS['text'], mirror=True, title='Daily new cases')

##### MODELS FOR DEATHS FORECASTING #####
pred_uk_deaths_ARIMA, conf_uk_deaths_ARIMA = construct_ARIMA(False)
pred_uk_deaths_LSTM, min_uk_deaths_LSTM, max_uk_deaths_LSTM = construct_LSTM(False)

index_pred_uk_deaths = pd.date_range(data_uk_deaths.index[-1], periods=22, freq='D', closed='right')
df_pred_uk_deaths = pd.DataFrame({'pred_ARIMA': pred_uk_deaths_ARIMA, 'min_ARIMA': conf_uk_deaths_ARIMA[:,0], 'max_ARIMA': conf_uk_deaths_ARIMA[:,1]})

df_pred_uk_deaths['pred_LSTM'] = pred_uk_deaths_LSTM
df_pred_uk_deaths['min_LSTM'] = min_uk_deaths_LSTM
df_pred_uk_deaths['max_LSTM'] = max_uk_deaths_LSTM

df_pred_uk_deaths['pred_ENSEMBLE'] = df_pred_uk_deaths[['pred_ARIMA', 'pred_LSTM']].mean(axis=1)
df_pred_uk_deaths['min_ENSEMBLE'] = df_pred_uk_deaths[['min_ARIMA', 'min_LSTM']].mean(axis=1)
df_pred_uk_deaths['max_ENSEMBLE'] = df_pred_uk_deaths[['max_ARIMA', 'max_LSTM']].mean(axis=1)

##### FIGURE FOR DEATHS FORECASTING #####
df_pred_uk_deaths = df_pred_uk_deaths.clip(lower=0)
average_uk_deaths = df_uk_deaths['newdeaths'].rolling(7,center=True).mean().round()

fig_uk_deaths = go.Figure()
fig_uk_deaths.add_bar(name='Recorded', x=df_uk_deaths['date'], y=df_uk_deaths['newdeaths'], marker={'color':COLORS['death_scatter']})
fig_uk_deaths.add_scatter(name='7-Day Average (Recorded)', x=df_uk_deaths['date'], y=average_uk_deaths, line={'color':COLORS['death'], 'width':3})
fig_uk_deaths.add_scatter(name='Max Predicted', x=index_pred_uk_deaths, y=df_pred_uk_deaths['max_ENSEMBLE'], line={'color':COLORS['pred']}, showlegend=False)
fig_uk_deaths.add_scatter(name='Min Predicted', x=index_pred_uk_deaths, y=df_pred_uk_deaths['min_ENSEMBLE'], line={'color':COLORS['pred'], 'width':1}, fill='tonexty', showlegend=False)
fig_uk_deaths.add_scatter(name='Predicted', x=index_pred_uk_deaths, y=df_pred_uk_deaths['pred_ENSEMBLE'], line={'color':COLORS['death_pred'], 'width':3})
fig_uk_deaths.add_scatter(name='Max Predicted', x=index_pred_uk_deaths, y=df_pred_uk_deaths['max_ARIMA'], line={'color':COLORS['pred']}, showlegend=False, visible=False)
fig_uk_deaths.add_scatter(name='Min Predicted', x=index_pred_uk_deaths, y=df_pred_uk_deaths['min_ARIMA'], line={'color':COLORS['pred'], 'width':1}, fill='tonexty', showlegend=False, visible=False)
fig_uk_deaths.add_scatter(name='Predicted', x=index_pred_uk_deaths, y=df_pred_uk_deaths['pred_ARIMA'], line={'color':COLORS['death_pred'], 'width':3}, visible=False)
fig_uk_deaths.add_scatter(name='Max Predicted', x=index_pred_uk_deaths, y=df_pred_uk_deaths['max_LSTM'], line={'color':COLORS['pred']}, showlegend=False, visible=False)
fig_uk_deaths.add_scatter(name='Min Predicted', x=index_pred_uk_deaths, y=df_pred_uk_deaths['min_LSTM'], line={'color':COLORS['pred'], 'width':1}, fill='tonexty', showlegend=False, visible=False)
fig_uk_deaths.add_scatter(name='Predicted', x=index_pred_uk_deaths, y=df_pred_uk_deaths['pred_LSTM'], line={'color':COLORS['death_pred'], 'width':3}, visible=False)
fig_uk_deaths.update_layout(updatemenus=DEATHS_UPDATE_MENUS, annotations=UK_DEATHS_EVENTS + FIXED_ANNOTATIONS, hovermode='x unified', font={'color':COLORS['text']},
                           plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['background'], legend={'orientation':'h', 'traceorder':'normal', 'xanchor':'center', 'x':0.5},
                            title='<b>Covid-19 Deaths Forecasting for the UK</b><br><sub>The forecast contains <i>three-week</i> prediction with <i>95%</i> confidence interval</sub>', title_x=0.5)
fig_uk_deaths.update_xaxes(range=[str(df_uk_deaths['date'].iloc[-1].date()), str(index_pred_uk_deaths[-1].date())],
                           showline=True, linecolor=COLORS['text'], mirror=True)
fig_uk_deaths.update_yaxes(showline=True, linecolor=COLORS['text'], mirror=True, title='Daily new deahts')
