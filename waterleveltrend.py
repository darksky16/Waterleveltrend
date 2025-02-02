import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import pymannkendall as mk
import os

# مسیر صحیح برای خواندن فایل در محیط Railway
csv_file = os.path.join(os.path.dirname(__file__), "combined_waterlevel.csv")
df = pd.read_csv(csv_file, encoding='utf-8-sig', low_memory=False)  # Handle mixed types and ensure Persian text compatibility

# Ensure that 'gregorian_date' is in datetime format
df['gregorian_date'] = pd.to_datetime(df['gregorian_date'], errors='coerce')

# Drop rows with invalid or missing dates
df = df.dropna(subset=['gregorian_date'])

# Fill missing values in numeric columns to avoid plotting issues
numeric_cols = ['sath_ab_jadid', 'taraz', 'sath-ab']
df[numeric_cols] = df[numeric_cols].fillna(0)

# Initialize Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Interactive Trends Visualization with Statistical Analysis", style={'textAlign': 'center'}),

    # Province Dropdown
    html.Div([
        html.Label("Select Provinces:"),
        dcc.Dropdown(
            id='province-filter',
            options=[{'label': province, 'value': province} for province in sorted(df['ostan'].dropna().unique())],
            value=None,  # Start with no province selected
            multi=True,
            placeholder="Select Province(s)"
        ),
    ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),

    # UTM Dropdown (Dynamically Populated)
    html.Div([
        html.Label("Select UTM Code(s):"),
        dcc.Dropdown(
            id='utm-filter',
            multi=True,  # Allow multiple selection for UTM codes
            placeholder="Select UTM Code(s)"
        ),
    ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),

    # Display Mahdoodeh
    html.Div([
        html.Label("Selected Mahdoodeh:"),
        html.Div(id='mahdoodeh-display', style={'padding': '10px', 'border': '1px solid #ccc', 'width': '50%'})
    ]),

    # Variable Dropdown
    html.Div([
        html.Label("Select Variable to Plot:"),
        dcc.Dropdown(
            id='variable-filter',
            options=[{'label': col, 'value': col} for col in numeric_cols],  # Add all relevant variables
            value='sath_ab_jadid',  # Default variable to plot
            placeholder="Select Variable to Plot"
        ),
    ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),

    # Graph
    dcc.Graph(id='trend-plot'),  # Interactive plot

    # Display Mann-Kendall and Sen's Slope Results
    html.Div([
        html.Label("Mann-Kendall and Sen's Slope Analysis:"),
        html.Div(id='trend-analysis-display', style={'padding': '10px', 'border': '1px solid #ccc', 'width': '80%'})
    ]),

    # Province-Level Summary
    html.Div([
        html.Label("Province-Level Summary:"),
        html.Div(id='province-summary-display', style={'padding': '10px', 'border': '1px solid #ccc', 'width': '80%'})
    ]),
])

# Callback to update UTM dropdown based on selected provinces
@app.callback(
    Output('utm-filter', 'options'),
    Input('province-filter', 'value')
)
def update_utm_dropdown(selected_provinces):
    if not selected_provinces:
        return []
    # Filter UTM codes based on selected provinces
    filtered_df = df[df['ostan'].isin(selected_provinces)]
    utm_options = [{'label': f"{utm} (Records: {len(filtered_df[filtered_df['UTM'] == utm])})", 'value': utm}
                   for utm in filtered_df['UTM'].dropna().unique()]
    return utm_options

# Callback to display mahdoodeh for selected UTM codes
@app.callback(
    Output('mahdoodeh-display', 'children'),
    Input('utm-filter', 'value')
)
def update_mahdoodeh_display(selected_utms):
    if not selected_utms:
        return "No UTM code(s) selected"
    mahdoodeh_list = df.loc[df['UTM'].isin(selected_utms), 'mahdoodeh'].dropna().unique()
    return ', '.join(mahdoodeh_list) if len(mahdoodeh_list) > 0 else "No mahdoodeh found"

# Callback to update plots, perform statistical analysis, and provide province-level summary
@app.callback(
    [Output('trend-plot', 'figure'),
     Output('trend-analysis-display', 'children'),
     Output('province-summary-display', 'children')],
    [Input('province-filter', 'value'),
     Input('utm-filter', 'value'),
     Input('variable-filter', 'value')]
)
def update_plot_and_analysis(selected_provinces, selected_utms, selected_variable):
    if not selected_provinces:
        return px.line(title="No Province Selected"), "No Province Selected", "No Province Selected"

    # Ensure a valid variable is selected
    if not selected_variable:
        selected_variable = numeric_cols[0]

    # Ensure UTM codes are not None
    if not selected_utms:
        selected_utms = []

    filtered_df = df[df['ostan'].isin(selected_provinces)]

    # Province-Level Summary
    utm_groups = filtered_df.groupby('UTM')
    significant_count = 0
    nonsignificant_count = 0
    total_utm = len(utm_groups)
    for utm, group in utm_groups:
        values = group[selected_variable].dropna().values
        if len(values) < 5:
            continue
        mk_result = mk.original_test(values)
        if mk_result.p < 0.05:
            significant_count += 1
        else:
            nonsignificant_count += 1

    # Calculate the ratio as a decimal
    ratio = significant_count / nonsignificant_count if nonsignificant_count > 0 else float("inf")

    province_summary = (f"Total UTM Codes: {total_utm}, "
                        f"Significant Trends: {significant_count} "
                        f"({(significant_count / total_utm) * 100:.2f}%), "
                        f"No Trends: {nonsignificant_count} "
                        f"({(nonsignificant_count / total_utm) * 100:.2f}%), "
                        f"Ratio (Significant:No Significant): {ratio:.2f} (as a number)")

    # Filter for selected UTM codes
    if selected_utms:
        filtered_df = filtered_df[filtered_df['UTM'].isin(selected_utms)]

    if filtered_df.empty:
        return px.line(title="No Data Available"), "No Data Available", province_summary

    # Plot the graph
    fig = px.line(
        filtered_df,
        x='gregorian_date',
        y=selected_variable,
        color='UTM',
        title=f"Trend of {selected_variable} by Province and UTM",
        labels={'gregorian_date': 'Gregorian Date', selected_variable: selected_variable},
        template='plotly'
    )
    fig.update_traces(mode='lines+markers')

    # UTM-level analysis
    analysis_results = []
    for utm in selected_utms:
        utm_data = filtered_df[filtered_df['UTM'] == utm][selected_variable].dropna().values
        if len(utm_data) < 5:
            analysis_results.append(f"UTM: {utm} - Insufficient Data")
            continue

        # Perform Mann-Kendall test
        mk_result = mk.original_test(utm_data)

        # Confidence interval estimation
        confidence_interval = f"{mk_result.slope_confidence_interval}" if hasattr(mk_result, 'slope_confidence_interval') else "N/A"

        # Trend and interpretation
        if mk_result.p < 0.05:
            trend = "Increasing" if mk_result.z > 0 else "Decreasing"
            interpretation = f"Positive Trend, Statistically Significant, {'Strong' if abs(mk_result.s) > 100 else 'Moderate'} Relationship"
        else:
            trend = "No Significant Trend"
            interpretation = "No Significant Trend, P-value exceeds 0.05"

        analysis_results.append(
            f"UTM: {utm}, Trend: {trend}, Sen's Slope: {mk_result.slope:.4f}, P-value: {mk_result.p:.4f}, "
            f"Z-score: {mk_result.z:.4f}, S-statistic: {mk_result.s:.4f}, Confidence Interval: {confidence_interval}, "
            f"Interpretation: ({interpretation})"
        )

    formatted_results = html.Div([
        html.Div(result, style={'margin-bottom': '10px'}) for result in analysis_results
    ])

    return fig, formatted_results, province_summary

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
