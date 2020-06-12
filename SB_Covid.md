---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Interactive Plots of Covid-19 in Santa Barbara County

As a resident of Santa Barbara County, I'm interested in keeping tabs on levels of Covid-19 infection in my area. Data are available [from the State](https://data.chhs.ca.gov/) and [county](https://publichealthsbc.org/status-reports/). However, I wasn't happy with the format in which the data were presented so I decided to do some visualizations. I'm most interested in tracking the level of current infections and the level on strain on local medical resources. 

```python
import re
import urllib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
```

```python
url = 'https://data.chhs.ca.gov/api/3/action/datastore_search?resource_id=6cd8d424-dfaa-4bdd-9410-a3d656e1176e&limit=50000'  
fileobj = json.loads(urllib.request.urlopen(url).read())['result']

df = pd.DataFrame(fileobj['records'])
df['Most Recent Date'] = pd.to_datetime(df['Most Recent Date'])
df.index = df['Most Recent Date']
df.index.name = 'Date'

sbdf = df.query("`County Name` == 'Santa Barbara'").sort_index()
name_map = {
    'COVID-19 Positive Patients': 'Hospitalized (Confirmed)',
    'ICU COVID-19 Positive Patients': 'ICU (Confirmed)',
    'Suspected COVID-19 Positive Patients': 'Hospitalized (Suspected)',
    'ICU COVID-19 Suspected Patients': 'ICU (Suspected)'
}
sbdf.rename(columns=name_map, inplace=True)

sbc = sbdf.copy()
sbc['Hospitalized (Confirmed)'] -= sbc['ICU (Confirmed)']
sbc['Hospitalized (Suspected)'] -= sbc['ICU (Suspected)']
sbc['Date'] = sbc.index

cm = plt.get_cmap('tab20')

def mpl_to_plotly(cmap):
    colors = ((np.array(c) * 255).astype('int') for c in cm.colors)
    pl_colorscale = [f'rgb{tuple(color)}' for color in colors]
    return pl_colorscale

pxtab20 = mpl_to_plotly(cm)
```

## Data from CA Health and Human Services

These data were downloaded from the [CHHS website](https://data.chhs.ca.gov/). They're downloadable in a consistant format from April onward. I've produced my own visualization that I prefer to the one [available on the CHHS website](https://public.tableau.com/views/COVID-19PublicDashboard/Covid-19Hospitals?:embed=y&:display_count=no&:showVizHome=no). You can also see a [full page version](hospitalized_sb.html).

This plot tracks the number of hospitalized Covid-19 patients in SB County over time, as well as cumulative Covid-19 deaths in SB County. It does not track infected patients who may be recovering at home. Patients designated here as "Confirmed" are those that have tested positive for Covid-19. Those designated as "Suspected" are pending test results. For more information, see the [CHHS website](https://data.chhs.ca.gov/dataset/california-covid-19-hospital-data-and-case-statistics).

```python
cm = plt.get_cmap('tab20')

def mpl_to_plotly(cmap, alpha=1.0):
    colors = ((np.array(c) * 255).astype('int') for c in cmap.colors)
    pl_colorscale = [f'rgba{tuple(color) + (alpha,)}' for color in colors]
    return pl_colorscale

pxtab20 = mpl_to_plotly(cm, 0.5)
pxtab10 = mpl_to_plotly(plt.get_cmap('tab10'), 0.5)
```

```python
sbcm = sbc.melt(value_vars=name_map.values(), id_vars='Date', var_name='Patient Category', value_name='Count')
pc_order = {
    'ICU (Confirmed)': '1',
    'ICU (Suspected)': '2',
    'Hospitalized (Confirmed)': '3',
    'Hospitalized (Suspected)': '4'
}
sbcm = pd.concat([sbcm.query("`Patient Category`==@cat") for cat in pc_order.keys()])
fig = px.area(sbcm, x='Date', y='Count', color='Patient Category',
              title="Hospitalized Covid-19 Patients in SB County", color_discrete_sequence=pxtab20)
tot = sbcm.groupby('Date').sum()
fig.add_trace(go.Scatter(x=tot.index, y=tot.Count, mode='lines', name='Total', line=dict(dash='dashdot', width=3, color='red')))
fig.add_trace(go.Scatter(x=sbdf.index, y=sbdf['Total Count Deaths'], mode='lines', name='Cumulative Deaths', 
                         line=dict(dash='dash', width=2, color='black')))
fig.show()
import plotly.io as pio
pio.write_html(fig, file='hospitalized_sb.html')
```

```python
death_per_confirmed = (sbdf['Total Count Deaths'].max() / sbdf['Total Count Confirmed'].max()) * 100
n_dead = sbdf['Total Count Deaths'].max().astype('int')
print(f"{n_dead} total deaths. {death_per_confirmed:.2f}% case fatality rate as of {sbdf.index.max().strftime('%B %-d, %Y')}.")
```

## Data From SB County Public Health Department

[These data](https://publichealthsbc.org/status-reports/) offer a more detailed look at the current status of Covid-19 in SB County. The numbers are broken down by geographic location, including separating the very large number of cases at the [Lompoc Federal Correctional Institution](https://en.wikipedia.org/wiki/Federal_Correctional_Institution,_Lompoc). However, they are not offered for download in any convenient format (at least, not that I've found) and the formatting on the website has not been consistent. I have written a script to scrape data from the site. It works with the format that was adopted on May 13th, so this data set only goes back that far.

```python
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

url = 'https://publichealthsbc.org/status-reports/'
req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})

page = urlopen(req).read()
soup = BeautifulSoup(page, 'html.parser')

status_containers = soup.findAll('div', class_='elementor-accordion-item')

def getDate(stat_con):
    date_text = stat_con.find('div', class_='elementor-tab-title').text.replace('\n', '').replace('as of ', '')
    return pd.to_datetime(str(date_text))

def getTable(stat_con):
    try:
        tab = stat_con.find('td', text='Active Cases').findParents('table')[0]
    except AttributeError:
        tab = None
    return tab

def getDataframe(stat_con):
    date = getDate(stat_con)
    tab_html = getTable(stat_con)
    df = pd.read_html(str(tab_html), header=0, index_col=0)[0]
    df = df.apply(lambda s: s.replace('—', '0')).apply(pd.to_numeric)
    df['Date'] = date
    df['Category'] = df.index.to_series()
    df.set_index('Date', inplace=True, drop=False)
    return df

def communityDF(stat_con_list):
    df_list = []
    for stat_con in stat_con_list:
        tab = getTable(stat_con)
        if tab:
            df_list.append(getDataframe(stat_con))
    return pd.concat(df_list).pivot(index='Date', columns='Category', values='Community')
        
tables = [getTable(sc) for sc in status_containers]
dfs = [getDataframe(sc) for sc in status_containers[:2]]

def shorten(s):
    pat = re.compile('([A-Z\s]+($|\s))')
    fa = pat.findall(s)
    s = fa[0][0].title() if fa else ''
    return s

def getGeogDF(soup):
    status_containers = soup.findAll('div', class_='elementor-accordion-item')
    for sc in status_containers:
        date = getDate(sc)
        try:
            tab = sc.find('strong', text='Geographic Area').findParents('table')[0]
        except AttributeError:
            continue
        df = pd.read_html(str(tab), header=0)[0]
        df['Date'] = date
        df['Date'] = df.Date.apply(lambda d: d.date()).apply(lambda d: d.strftime('%b %d'))
        df['Geographic Area'] = df['Geographic Area'].apply(shorten)
        df = df.query("`Geographic Area` != ''")
    return df

def getDailyCases(soup):
    status_containers = soup.findAll('div', class_='elementor-accordion-item')
    df_list = []
    for sc in status_containers:
        date = getDate(sc)
        try:
            tab = sc.find('strong', text='Geographic Area').findParents('table')[0]
        except AttributeError:
            continue
        df = pd.read_html(str(tab), header=0)[0]
        df['Date'] = date
        df['date_label'] = df.Date.apply(lambda d: d.date()).apply(lambda d: d.strftime('%b %d'))
        df['Geographic Area'] = df['Geographic Area'].apply(shorten)
        df = df.query("`Geographic Area` != ''")
        keep_these = ['Date', 'Daily Cases', 'Geographic Area']
        df_list.append(df[keep_these])
    df = pd.concat(df_list)
    df = pd.pivot(df, index='Date', columns='Geographic Area', values='Daily Cases')
    return df

def getTotalConfirmed(soup):
    status_containers = soup.findAll('div', class_='elementor-accordion-item')
    df_list = []
    for sc in status_containers:
        date = getDate(sc)
        try:
            tab = sc.find('strong', text='Geographic Area').findParents('table')[0]
        except AttributeError:
            continue
        df = pd.read_html(str(tab), header=0)[0]
        df['Date'] = date
#         df['Date'] = df.Date.apply(lambda d: d.date()).apply(lambda d: d.strftime('%b %d'))
        df['Geographic Area'] = df['Geographic Area'].apply(shorten)
        df = df.query("`Geographic Area` != ''")
        keep_these = ['Date', 'Total Confirmed Cases', 'Geographic Area']
        df_list.append(df[keep_these])
    df = pd.concat(df_list)
    df = pd.pivot(df, index='Date', columns='Geographic Area', values='Total Confirmed Cases')
    return df
```

```python
cdf = communityDF(status_containers)
cdf['Date'] = cdf.index
cdf = pd.DataFrame(cdf.to_records(index=False)).sort_values('Date')
plot_cols = ['Pending Information', 'Recovering at Home', 'Recovering in Hospital', 'Recovering in ICU', 'Date']
cdfm = cdf[plot_cols].melt(value_vars=['Pending Information', 'Recovering at Home', 'Recovering in Hospital', 'Recovering in ICU'],
                           id_vars='Date', var_name='Category', value_name='Count')
cat_order = [
    'Recovering in ICU',
    'Recovering in Hospital',
    'Recovering at Home',
    'Pending Information'
]
cdfm = pd.concat([cdfm.query("`Category`==@cat") for cat in cat_order])
cdfm.Category.unique()

fig = px.bar(cdfm, x='Date', y='Count', color='Category', color_discrete_sequence=pxtab10,
            title="Active Covid-19 Cases in SB County (Excluding Lompoc Prison)")
tot = cdfm.query("Category != 'Pending Information'").groupby('Date').sum()
fig.add_trace(go.Scatter(x=tot.index, y=tot.Count, mode='lines', name='Total Confirmed Cases', line=dict(dash='dashdot', width=3, color='red')))
fig.show()
```

### Additional Plots from County Data

These are some additional non-interactive plots based on the SB County data. They explore the geographic break down of Covid-19 cases in SB County. Please note that there are gaps in the x-axis for days where numbers were not reported. On the plot above, these gaps are clearly visible. On the plots below, the gaps in reporting still exist but they are more difficult to see.

```python
dcdf = getDailyCases(soup).astype('float')
ax = dcdf.plot.bar(stacked=True, figsize=(13, 6), title='SB County: Daily New Cases by Area')
foo = ax.set_ylabel('Cases')
foo = plt.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", borderaxespad=0)
```

```python
tcdf = getTotalConfirmed(soup).astype('float')
ax = tcdf.plot.bar(stacked=True, figsize=(13, 6), title='SB County: Total Confirmed Cases by Area')
foo = ax.set_ylabel('Cases')
foo = plt.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", borderaxespad=0)
```

```python
ax = tcdf.drop("Federal Prison In Lompoc", axis=1).plot.bar(stacked=True, figsize=(13, 6), title='SB County: Total Confirmed Cases by Area (Excluding Prison)')
foo = ax.set_ylabel('Cases')
foo = plt.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", borderaxespad=0)
```

# Caveats!

No one is paying me to do this. I do have a PhD, but it's in Marine Science, so I am in no way qualified to give out public health advice. I'm not trying to. I'm not responsible for the quality of the data. I'm just trying to visualize some one else's data set. ...and I did it as quickly as possible, so I may have even done that wrong. Feel free to take a look at [the code I wrote](https://github.com/jkibele/SB_Covid/blob/master/SB_Covid.md) and offer helpful suggestions or just tell me I'm dumb and did stuff wrong.

```python
%%capture
!jupyter nbconvert SB_Covid.ipynb --to html --no-input --output index.html
```
