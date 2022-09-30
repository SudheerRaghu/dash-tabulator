import dash, re, platform, os, colorlover, json
import pandas as pd, requests, numpy as np
import redis
from dash import dash_table, ctx
import dash.dash_table.FormatTemplate as FormatTemplate
from dash.dependencies import Input, Output, State, ClientsideFunction
from dash.exceptions import PreventUpdate
from dash_bootstrap_templates import load_figure_template
from dash import dcc
from dash import html
import plotly.express as px
from loguru import logger
import dash_bootstrap_components as dbc
from flask_caching import Cache
import dash_tabulator, dash_auth
from dash_extensions.javascript import Namespace

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
bs = ["https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"]
os.environ['no_proxy'] = '*'

dbc_css = ("https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css")
external_scripts = [{'src': 'https://oss.sheetjs.com/sheetjs/xlsx.full.min.js'}]
load_figure_template(["bootstrap","spacelab"])

debug_state = True
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css], suppress_callback_exceptions=True,
                title='ARA Valuation Database', external_scripts=external_scripts)

ns = Namespace("dash_clientside", "clientside")

def remove_chars(s):
    return re.findall("([0-9]+[.]+[0-9]+)", s)[0]

# df = px.data.tips()
def load_data(df=pd.DataFrame()):
    logger.info('Loading data...')
    # df = pd.read_csv('./valuation_results.csv')
    # if df.empty:
    #     df = pd.read_excel('sample_data_sunburst.xlsx')
    # df = pd.DataFrame(smd_config)
    logger.info(df.columns)
    # df['source'] = np.where(df['root_url'] == 'https://www.lazada.sg', 'LAZADA',
    #                         np.where(df['root_url'] == 'https://www.amazon.sg', 'AMAZON', 'SHOPEE'))
    df['price'] = df['price'].astype(str).apply(remove_chars)
    df['price'] = df['price'].astype(float)
    # df.index.name = 'S No'
    # df.index += 1
    # df.reset_index(inplace=True)
    rename_cols = {'tag': 'Tag',
                   'product_name': 'Product Name',
                   'price': 'Price',
                   'root_url': 'Root URL',
                   'source_url': 'Source URL',
                   'country': 'Country',
                   'source': 'Source',
                   'category': 'Category',
                   'sub_category': 'Sub-Category',
                   'serial_num': 'S No',
                   'data_source': 'Source',
                   'uom': "UOM",
                   'quantity': "Quantity"}
    df.rename(columns=rename_cols, inplace=True)
    df.sort_values(by = 'S No', ascending=True, inplace=True)
    df['count'] = 1 # df.groupby('Tag')['Tag'].transform('count')
    # df['price_max'] = df['Price'] # df.groupby('Tag')['Price'].transform('max')
    # df['price_mean'] = df['Price'] # df.groupby('Tag')['Price'].transform('mean')
    # df['Root URL'] = df['Root URL'].apply(lambda x: "{}{}{}{}{}{}".format('[', x, ']', '(', x, ')'))
    # df['Source URL'] = df['Source URL'].apply(lambda x: "{}{}{}{}{}{}".format('[', 'URL', ']', '(', x, ')'))
    # df['count'] = df.groupby('Sub-Category')['Sub-Category'].transform('count')
    # for col in df.columns:
    #     if df[col].dtypes == 'O' and col not in ['Source URL']:
    #         df[col] = df[col].str.upper()
    # logger.info(df['Source URL'][0])
    return df


# df = load_data()
# path = ["day", "time", "sex"]
total_path = ['Source', 'Tag', 'Category', 'Sub-Category']
ordered_path = [1, 2, 3, 4]
value_path = [x + '_' + str(y) for x, y in zip(total_path, ordered_path)]


# path = ['Tag', 'Product Name']

def path_check_list():
    checklist = html.Div(
        [
            dbc.Label("Select Path:"),
            dbc.Checklist(
                id="path_select",
                # options=[{"label": i, "value": i} for i in total_path],
                options=[{"label": i, "value": j} for i, j in zip(total_path, value_path)],
                value=value_path[-2:],
                inline=True

            ),
        ],
        className="mb-4",
    )
    return checklist

def sunburst_chart(df, path=None):
    logger.info(path)
    if path is None:
        path = total_path
    # logger.info(df.columns)
    fig = px.sunburst(df, path=path, values='count',
                      # hover_data=['product_name', 'price'],
                      color='Price', range_color=[0, df['Price'].max()],
                      # color_continuous_scale='RdBu',
                      color_continuous_midpoint=df['Price'].mean(),
                      height=600,
                      template="bootstrap")
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20)
        # paper_bgcolor="LightSteelBlue",
    )
    return fig

def dash_tabulator_table(df):
    cols_type = []
    # logger.info(df.dtypes)
    # df['S.No']=df['S.No'].astype(float)
    for col in df.columns:
        if col not in ['score', 'count', 'price_max', 'price_mean', 'Root URL']:
            if col in ['Root URL', 'Source URL']:
                # cols_type.append({"name": col, "id": col, 'type': 'text', 'presentation': 'markdown'})
                cols_type.append({"title": col, "field": col, "hozAlign": "left", "headerFilter": True,
                                  "headerFilterParams":{"initial":""}, "formatter": "link",
                                  "formatterParams": {"label": "URL", "target":"_blank"}
                                  })
            elif col in ['Price']:
                cols_type.append({
                                "title": col,
                                "field": col,
                                "hozAlign": "right",
                                "formatter": "money",
                                "formatterParams": {"precision": 2},
                                "headerFilter": True,
                                "headerFilterParams": {"initial": ""},
                                # "editor":"range",
                                "editorParams":{"min": 0, "max": df['Price'].max(), "step": 5},
                                "headerFilterFunc": "<=",
                                "headerFilterPlaceholder": "<="
                                })
            elif col in ['Product Name']:
                cols_type.append({"title": col, "field": col, "hozAlign": "left", "headerFilter": True,
                                  "headerFilterParams": {"initial": ""}, "width": 300})
            else:
                cols_type.append({"title": col, "field": col, "hozAlign": "left", "headerFilter": True,
                                  "headerFilterParams":{"initial":""}})
    options = {"pagination": "local","paginationSize":20,"clearHeaderFilter":"true",
               # "resizableRows":"false",
               "resizableColumns":"false",
               "initialFilter":[], "filterMode":"local",
               "frozenRows":1,
               "dataFiltered": ns("clear_filters"),
               "initialHeaderFilter": [],
               # "clearFilter":"true",
               "movableColumns":"false",
               "layout":"fitDataTable",
               "downloadConfig":{"downloadRowRange":"all", "columnHeaders":"true"}}
    downloadButtonType = {"css": "btn btn-primary", "text": "Export", "type": "xlsx"}
    clearFilterButtonType = {"css": "btn btn-outline-dark", "text": "Clear Filters"}
    # clearFilterButtonType = html.Div([dbc.Button("Clear Filters", color="primary", outline=True)])
    table_layout = html.Div(
        [
            dash_tabulator.DashTabulator(
                id="table",
                theme="tabulator_bootstrap4",
                columns=cols_type,
                data=df.to_dict("records"),
                initialHeaderFilter=[],
                dataFiltering=[],
                # dataFiltered = False,
                options=options,
                downloadButtonType=downloadButtonType,
                clearFilterButtonType=clearFilterButtonType,

            )])
    return table_layout

def create_form():
    form = html.Div([html.H4(children='Valuation Risk Analytics', style={'textAlign': 'center'},
                             className="bg-primary text-white p-2 mb-2 text-center"),
                     html.Div(
                         [
                             dbc.Container([
                     dbc.Accordion([dbc.AccordionItem(
                         dbc.Form([
                         dbc.Row(
                             [
                                 dbc.Label("Country", width=1),
                                 # dbc.Col(
                                     # drop_downs(),
                                 dbc.Input(type="text", placeholder="Enter Country - Ex: Singapore, India etc",
                                               id='country', value='Singapore',
                                               style={"margin-left": "5px", "margin-right": "15px",
                                                      "margin-top": "15px", "margin-bottom": "20px", "width": "15%"},
                                               required=True, debounce=True, autocomplete=True),
                                     # className="me-3",

                                 # ),
                                 dbc.Label("Product Description", width=2),
                                 # dbc.Col(
                                 dbc.Input(type="text", placeholder="Enter Product Description", id='desc',
                                               style={"margin-left": "10px", "margin-right": "15px",
                                                      "margin-top": "15px", "margin-bottom": "20px", "width": "55%"},
                                               required=True, debounce=True),
                                     # className="me-3",

                                 # )
                             ],
                             className="g-2",
                         ),
                         dbc.Row(
                             [
                                 dbc.Label("HS Code", width=1),
                                 # dbc.Col(
                                 dbc.Input(type="text", placeholder="Enter HS Code", id='hscode',
                                           style={"margin-left": "5px", "margin-right": "15px",
                                                  "margin-top": "15px", "margin-bottom": "10px", "width": "15%"},
                                               debounce=True),
                                     # className="me-3",
                                     # ),
                                 dbc.Label("Exclude Keywords", width=2),
                                 # dbc.Col(
                                 dbc.Input(type="text", placeholder="Enter Exclude Keywords", id='exclude',
                                           value='None',
                                           style={"margin-left": "10px", "margin-right": "15px",
                                                  "margin-top": "15px", "margin-bottom": "10px", "width": "55%"},
                                           debounce=True),

                                 # dbc.Button("Submit", color="primary", active=True, outline=True,id='button', size="sm",
                                 #            style={"margin-top": "15px", "margin-bottom": "10px", "width":"10%"}),
                                     # className="me-3",

                                 # )
                             ],
                             className="g-3"),
                         dbc.Row([dbc.Button("Submit", color="primary", active=True, outline=True,id='button', size="sm",
                                            style={"margin-left": "80%","margin-top": "15px", "margin-bottom": "5px", "width":"10%"})]),
                     ],
                         id='form-submit'
                     ), title='Search')], id='search-accordion',start_collapsed=False, persistence =True)], fluid=True,className="dbc")
                     ])
                     ])

    return form


def radio_button():
    button_group = dbc.RadioItems(
        id="radios",
        className="btn-group",
        inputClassName="btn-check",
        labelClassName="btn btn-outline-primary",
        labelCheckedClassName="active",
        options=[
            {"label": "Hierarchical", "value": 1},
            {"label": "Filter", "value": 2}
        ],
        value=2,
    )
    return button_group

def alert_proc_ui(error_type):
    if error_type == 'ERROR':
        error_msg = "Error processing the request. !"
    elif error_type == 'EMPTY':
        error_msg = "No Data Available. !"
    elif error_type == 'SERVER_DOWN':
        error_msg = 'Server Down. Please try later. !'
    else:
        error_msg = 'Unknown Error !'
    alerts = html.Div(
        [
            dbc.Alert(error_msg, color="danger", dismissable=True, fade=True, duration=4000),
        ]
    )
    return alerts


app.layout = dbc.Container(html.Div([
    html.Div(create_form(),
             style={
                 'background-image': ['url("/assets/cl-gets.png")'],
                 'background-repeat': ['no-repeat'],
                 'background-position': ['right top'],
                 # 'background-size': '150px 100px'
             }
             ),
    html.Div(id='graph-output', style={'padding': '45px 0px 0px 0px'}),
    dcc.Store(id='global-df'),
    dcc.Store(id='global-chart-df')
]), fluid=True, className="dbc")


def graph_layout_data(df, input_json):
    # logger.info(df.head())
    inputs = "{}, {}, {} and {}".format(input_json['country'], input_json['product_name'], input_json['hs_code'],
                                        input_json['exclude_words'])
    graph_layout = html.Div([
        html.Div(
            [
                dbc.Alert(['Search results for ', html.A(inputs, href='#', className='alert-link')],
                          color="success", dismissable=True, fade=True, duration=10000),
            ]
        ),
        dbc.Accordion([dbc.AccordionItem(
            html.Div(
                [
                    html.H4(id="title", style={'textAlign': 'center', 'padding': '180px 360px 30px 360px'},
                            className="bg-primary text-white p-2 mb-2 text-center"),
                    dbc.Row([
                        dbc.Col(html.Div(radio_button()), className="me-3", width=4),
                        dbc.Col(html.Div(path_check_list(), className="me-3"))
                    ]),
                    dbc.Spinner(children=dcc.Graph(
                        id="sunburst",
                        figure=sunburst_chart(df)),
                        size='lg', color='success', type='grow', fullscreen=False)
                ]), title='Chart'),],id='graph-accordion', start_collapsed=False, persistence =True),

        dbc.Accordion([dbc.AccordionItem(
            # html.Div(dbc.Spinner(dash_data_table(df), id='full-table', size='lg', color='success', type='grow', fullscreen=False))
            html.Div(dbc.Col(dash_tabulator_table(df), id='full-table', width=10))
            , title='Table'),
        ],
        id='table-accordion',start_collapsed=False, persistence =True)])
        # ,
        # fluid=True, className="dbc")
    return graph_layout

@app.callback(
    Output("search-collapse", "is_open"),
    [Input("button-collapse", "n_clicks")],
    [State("search-collapse", "is_open")],
)
def toggle_search_bar(n_left, is_open):
    if n_left:
        return not is_open
    return is_open


@app.callback([Output('sunburst', 'figure'), Output('path_select', 'value')],
              inputs=[Input('path_select', 'value')],
              state=[State('global-df', 'data')])
def update_sunburst_chart(selected_path, global_data):
    org_path = [i.split('_', 1)[0] for i in selected_path]
    ord_path = [int(i.split('_', 1)[1]) for i in selected_path]
    path = [x for _, x in sorted(zip(ord_path, org_path))]
    logger.info(path)
    df = pd.read_json(global_data, orient='split')
    df_chart_data = df[total_path + ['count', 'Price']]
    logger.info(df_chart_data.shape)
    return sunburst_chart(df_chart_data, path), selected_path


@app.callback(
    [Output("full-table", "children"), Output("title", "children")],
    [Input("button", "n_clicks"), Input("sunburst", "clickData"),Input("radios", "value"), Input('global-df', 'data'),
     Input('path_select', 'value')],
)
def update_table(n_clicks, clickData,filter_val, global_data, selected_path):
    # global df
    org_path = [i.split('_', 1)[0] for i in selected_path]
    ord_path = [int(i.split('_', 1)[1]) for i in selected_path]
    path = [x for _, x in sorted(zip(ord_path, org_path))]
    df = pd.read_json(global_data, orient='split')
    # path = selected_path
    logger.info(filter_val)
    if n_clicks:
        click_path_title, click_path = "ALL", "ALL"
        root = False
        data = df.to_dict(orient='records')
        if clickData and ctx.triggered_id == 'sunburst':
            logger.info(clickData)
            logger.info(clickData["points"][0]["id"])
            click_path = clickData["points"][0]["id"].split("/")
            click_path_title = ' | '.join(clickData["points"][0]["id"].split("/"))
            # entry = list(clickData['points'][0].get('entry'))
            percentEntry = (clickData["points"][0]).get("percentEntry")
            # click_path = list(set(click_path) - set(entry))
            # if percentEntry == 1:
            #     click_path = [x for x in click_path_temp if x not in entry]
            # else:
            #     click_path = click_path_temp
            selected = dict(zip(path, click_path))
            logger.info('{}, {}, {}'.format(path, click_path, selected))
            current_path = [x for x in clickData["points"][0].get("currentPath").split("/") if x != '']
            if "Sub-Category" in selected:
                if filter_val == 1:
                    dff = df.loc[(df[list(selected)] == pd.Series(selected)).all(axis=1)] if percentEntry < 1 else df.loc[(df[list(selected)] == pd.Series(selected)[:-1]).all(axis=1)]
                else:
                    dff = df[(df["Sub-Category"] == selected["Sub-Category"])] if percentEntry < 1 else df
                    click_path_title = ' | '.join(['All Sources', 'All Tags', 'All Categories', selected['Sub-Category']]) if percentEntry < 1 else ' | '.join(['All Sources', 'All Tags', 'All Categories'])
            elif "Category" in selected:
                if filter_val == 1:
                    if percentEntry < 1:
                        dff = df.loc[(df[list(selected)] == pd.Series(selected)).all(axis=1)]
                    else:
                        click_path_title = current_path if len(current_path) == 1 else ' | '.join(current_path)
                        click_path_title = 'ALL' if click_path_title == '' else click_path_title
                        selected_unfilter = dict(zip(path, current_path))
                        logger.info('{},{}'.format(click_path_title, selected_unfilter))
                        dff = df.loc[(df[list(selected_unfilter)] == pd.Series(selected_unfilter)).all(axis=1)]
                else:
                    if percentEntry < 1:
                        dff = df[(df["Category"] == selected["Category"])]
                        click_path_title = ' | '.join(['All Sources', 'All Tags', selected['Category']])
                    else:
                        # selected_unfilter = dict(zip(path[-1].split(), current_path[-1].split('|')))
                        selected_unfilter = pd.Series(selected)[(pd.Series(selected) == current_path[-1])].to_dict() if current_path != [] else {}
                        logger.info('{},{}'.format(current_path, selected_unfilter))
                        dff = df.loc[(df[list(selected_unfilter)] == pd.Series(selected_unfilter)).all(axis=1)]
                        click_path_title = ' | '.join(['All Sources', current_path[-1]]) if current_path != [] else ' | '.join(['All Sources', 'All Tags'])
            elif "Tag" in selected:
                if filter_val == 1:
                    if percentEntry < 1:
                        dff = df.loc[(df[list(selected)] == pd.Series(selected)).all(axis=1)]
                    else:
                        click_path_title = current_path
                        selected_unfilter = dict(zip(path, current_path))
                        click_path_title = 'ALL' if click_path_title == [] else click_path_title
                        logger.info('{},{}'.format(click_path_title, selected_unfilter))
                        dff = df.loc[(df[list(selected_unfilter)] == pd.Series(selected_unfilter)).all(axis=1)]
                else:
                    if percentEntry < 1:
                        dff = df[(df["Tag"] == selected["Tag"])]
                        click_path_title = ' | '.join(['All Sources', selected['Tag']])
                    else:
                        selected_unfilter = dict(zip(path, current_path))
                        logger.info(selected_unfilter)
                        dff = df.loc[(df[list(selected_unfilter)] == pd.Series(selected_unfilter)).all(axis=1)]
                        click_path_title = ' | '.join(['ALL' if selected.get("Source",'') == '' else selected.get("Source",'')])
            else:
                dff = df[(df["Source"] == selected["Source"])]
                root = True

            data = dff.to_dict("records")

            if root and percentEntry == 1:
                data = df.to_dict("records")
                click_path_title, click_path = "ALL", "ALL"

        title = f"{''.join(click_path_title)}"

        # table_changed = dash_data_table(pd.DataFrame.from_dict(data, orient='columns'))
        table_changed = dash_tabulator_table(pd.DataFrame.from_dict(data, orient='columns'))
        # table_changed = dash_dbc_table(pd.DataFrame.from_dict(data, orient='columns'))
        # return data, title
        return table_changed, title
    else:
        raise PreventUpdate


@app.callback(output=[Output('graph-output', 'children') ,Output('country', 'value'), Output('desc', 'value'),
                      Output('hscode', 'value'), Output('exclude', 'value'), Output('global-df', 'data'),
                      Output('global-chart-df', 'data')],
              inputs=[Input('button', 'n_clicks')],
              state=[State('country', 'value'), State('desc', 'value'), State('hscode', 'value'),
                     State('exclude', 'value')])
def show_graph(n_clicks, country, desc, hscode, exclude_key):
    # global df
    if n_clicks:
        # logger.info('Button clicked')
        input_json = {}
        input_json['hs_code'] = hscode
        input_json['country'] = country
        input_json['exclude_words'] = exclude_key
        input_json['product_name'] = desc
        logger.info(input_json)
        try:
            df = load_data(pd.read_excel('sample_data_sunburst.xlsx'))
            logger.info(df.columns)
            df.fillna('Others', inplace=True)
            df_chart_data = df[total_path + ['count', 'Price']]
            # df_chart_data.loc['count'] = df.loc['count']
            # df_chart_data.loc['price_max'] = df.loc['price_max']
            # df_chart_data.loc['price_mean'] = df.loc['price_mean']
            df_chart_data.drop_duplicates(inplace=True)
            logger.info('Processing data completed...')
        except Exception as e:
            logger.exception('Error processing - {}'.format(str(e)))
            return alert_proc_ui('ERROR'), 'Singapore', '', '', 'None', '', ''

        graph_layout = graph_layout_data(df, input_json)
        return graph_layout,'Singapore', '', '', 'None', df.to_json(date_format='iso',orient='split'), df_chart_data.to_json(date_format='iso', orient='split')
    else:
        raise PreventUpdate

if __name__ == "__main__":
    app.run_server(debug=debug_state)
elif __name__ != '__main__':
    app = app.server
