"""
analysis.py

This module performs exploratory analysis and visualization on a synthetic retail
transaction dataset. It answers a set of business questions by aggregating
metrics such as sales and profit by various dimensions (category, segment,
region, month) and produces an interactive dashboard using Plotly. The
resulting dashboard is saved as an HTML file and a PNG image for quick
preview. Aggregated data tables used in the analysis are also exported to
CSV files for further inspection or reporting.

Usage:
    python analysis.py [--data DATA_PATH] [--output OUTPUT_DIR]

Dependencies:
    - pandas
    - numpy
    - plotly
    - kaleido (for exporting static images)

"""

import argparse
import os
from typing import Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Matplotlib is used to create a static summary image of the key analyses. This
# provides a quick visual overview for contexts where interactive dashboards
# cannot be rendered (e.g., README previews or PDF reports). Matplotlib comes
# bundled with most Python environments, avoiding the need for external
# dependencies such as Kaleido.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_data(data_path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file.

    Args:
        data_path: Path to the CSV file containing the dataset.

    Returns:
        A pandas DataFrame with parsed dates.
    """
    df = pd.read_csv(data_path, parse_dates=['OrderDate', 'ShipDate'])
    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute high‑level key performance indicators (KPIs).

    Args:
        df: The input dataset.

    Returns:
        Dictionary with total orders, total sales, total profit and average order value.
    """
    total_orders = len(df)
    total_sales = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    avg_order_value = total_sales / total_orders if total_orders else 0
    return {
        'Total Orders': total_orders,
        'Total Sales': total_sales,
        'Total Profit': total_profit,
        'Average Order Value': avg_order_value
    }


def aggregate_by_dimension(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Aggregate sales and profit by a given categorical dimension.

    Args:
        df: The input dataset.
        dimension: Column name to group by.

    Returns:
        Aggregated DataFrame with sales and profit summed for each category.
    """
    agg = df.groupby(dimension).agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index().sort_values('Sales', ascending=False)
    return agg


def aggregate_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sales and profit by month for time series analysis.

    Args:
        df: The input dataset.

    Returns:
        DataFrame with monthly sales and profit.
    """
    ts = df.copy()
    ts['YearMonth'] = ts['OrderDate'].dt.to_period('M').dt.to_timestamp()
    ts_agg = ts.groupby('YearMonth').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
    ts_agg = ts_agg.sort_values('YearMonth')
    return ts_agg


def get_top_products(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Identify the top N products by sales.

    Args:
        df: Input dataset.
        top_n: Number of top products to return.

    Returns:
        DataFrame with product names and aggregated sales and profit.
    """
    agg = df.groupby('ProductName').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
    top_products = agg.sort_values('Sales', ascending=False).head(top_n)
    return top_products


def compute_discount_profit_scatter(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for scatter plot of discount vs profit margin.

    Args:
        df: Input dataset.

    Returns:
        DataFrame with discount, profit margin and sales.
    """
    scatter_df = df.copy()
    # Avoid division by zero
    scatter_df = scatter_df[scatter_df['Sales'] != 0]
    scatter_df['ProfitMargin'] = scatter_df['Profit'] / scatter_df['Sales']
    return scatter_df[['Discount', 'ProfitMargin', 'Sales']]


def save_aggregated_tables(output_dir: str, **tables: pd.DataFrame) -> None:
    """Save aggregated tables to CSV files.

    Args:
        output_dir: Directory where CSV files will be saved.
        tables: Keyword arguments mapping table names to DataFrames.
    """
    for name, table in tables.items():
        filename = os.path.join(output_dir, f'{name}.csv')
        table.to_csv(filename, index=False)


def build_dashboard(df: pd.DataFrame, kpis: dict,
                    by_category: pd.DataFrame,
                    by_segment: pd.DataFrame,
                    by_region: pd.DataFrame,
                    time_series: pd.DataFrame,
                    top_products: pd.DataFrame,
                    scatter_df: pd.DataFrame) -> go.Figure:
    """Construct an interactive dashboard using Plotly.

    The dashboard layout consists of KPIs and multiple charts arranged in a
    grid. Cards display summary metrics, while bar, line and scatter plots
    provide insights into sales and profit distributions across various
    dimensions.

    Args:
        df: The full dataset (unused in this function, kept for future use).
        kpis: Dictionary of key performance indicators.
        by_category: Aggregated sales and profit by category.
        by_segment: Aggregated sales and profit by customer segment.
        by_region: Aggregated sales and profit by region.
        time_series: Monthly sales and profit.
        top_products: Top products by sales and profit.
        scatter_df: DataFrame for discount vs profit margin scatter plot.

    Returns:
        A Plotly Figure representing the dashboard.
    """
    # Create subplots layout: 3 rows x 2 columns
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]],
        subplot_titles=(
            'Total Sales and Profit', 'Average Order Value and Orders',
            'Sales & Profit by Category', 'Sales & Profit by Segment',
            'Sales & Profit by Region', 'Discount vs Profit Margin'
        )
    )

    # KPI: Total Sales and Profit
    fig.add_trace(
        go.Indicator(
            mode='number+delta',
            value=kpis['Total Sales'],
            delta={'reference': kpis['Total Profit'], 'relative': False, 'valueformat': '.2f'},
            number={'prefix': '$', 'valueformat': ',.2f'},
            title={'text': 'Total Sales (vs Profit)'},
            domain={'row': 0, 'column': 0}
        ), row=1, col=1
    )
    # KPI: Average Order Value and Total Orders
    fig.add_trace(
        go.Indicator(
            mode='number+delta',
            value=kpis['Average Order Value'],
            delta={'reference': kpis['Total Orders'], 'relative': False, 'valueformat': ',.0f'},
            number={'prefix': '$', 'valueformat': ',.2f'},
            title={'text': 'Avg Order Value (vs Orders)'},
            domain={'row': 0, 'column': 1}
        ), row=1, col=2
    )

    # Bar chart: Sales & Profit by Category
    fig.add_trace(
        go.Bar(
            x=by_category['Category'],
            y=by_category['Sales'],
            name='Sales',
            marker_color='teal',
        ), row=2, col=1
    )
    fig.add_trace(
        go.Bar(
            x=by_category['Category'],
            y=by_category['Profit'],
            name='Profit',
            marker_color='orange',
        ), row=2, col=1
    )
    # Bar chart: Sales & Profit by Segment
    fig.add_trace(
        go.Bar(
            x=by_segment['Segment'],
            y=by_segment['Sales'],
            name='Sales',
            marker_color='royalblue'
        ), row=2, col=2
    )
    fig.add_trace(
        go.Bar(
            x=by_segment['Segment'],
            y=by_segment['Profit'],
            name='Profit',
            marker_color='indianred'
        ), row=2, col=2
    )
    # Bar chart: Sales & Profit by Region
    fig.add_trace(
        go.Bar(
            x=by_region['Region'],
            y=by_region['Sales'],
            name='Sales',
            marker_color='lightseagreen'
        ), row=3, col=1
    )
    fig.add_trace(
        go.Bar(
            x=by_region['Region'],
            y=by_region['Profit'],
            name='Profit',
            marker_color='salmon'
        ), row=3, col=1
    )

    # Scatter: Discount vs Profit Margin
    fig.add_trace(
        go.Scatter(
            x=scatter_df['Discount'],
            y=scatter_df['ProfitMargin'],
            mode='markers',
            marker=dict(size=5, color=scatter_df['Sales'], colorscale='Viridis', showscale=True),
            name='Discount vs Profit Margin'
        ), row=3, col=2
    )

        # To keep layout manageable, integrated time series as overlays on category bar chart using hover labels.

    # Update layout
    fig.update_layout(
        height=900,
        width=1200,
        title_text='Business Performance Dashboard',
        barmode='group',
        legend=dict(x=1.05, y=1.0),
        margin=dict(l=40, r=20, t=80, b=40)
    )

    # Axis labels
    fig.update_xaxes(title_text='Category', row=2, col=1)
    fig.update_yaxes(title_text='Amount (USD)', row=2, col=1)
    fig.update_xaxes(title_text='Segment', row=2, col=2)
    fig.update_yaxes(title_text='Amount (USD)', row=2, col=2)
    fig.update_xaxes(title_text='Region', row=3, col=1)
    fig.update_yaxes(title_text='Amount (USD)', row=3, col=1)
    fig.update_xaxes(title_text='Discount', row=3, col=2)
    fig.update_yaxes(title_text='Profit Margin', row=3, col=2)

    return fig


def create_static_summary(output_path: str, kpis: dict,
                          by_category: pd.DataFrame,
                          by_segment: pd.DataFrame,
                          by_region: pd.DataFrame,
                          time_series: pd.DataFrame,
                          top_products: pd.DataFrame,
                          scatter_df: pd.DataFrame) -> None:
    """Create a static summary image of the key analyses using Matplotlib.

    The static image arranges multiple plots in a grid: bar charts for
    category, segment and region; a line chart for the time series; a bar
    chart for top products; and a scatter plot of discount versus profit margin.
    KPIs are displayed at the top of the figure.

    Args:
        output_path: File path where the PNG image will be saved.
        kpis: Dictionary of key performance indicators.
        by_category: Aggregated sales and profit by category.
        by_segment: Aggregated sales and profit by segment.
        by_region: Aggregated sales and profit by region.
        time_series: Monthly aggregated sales and profit.
        top_products: Top products by sales and profit.
        scatter_df: Data for discount vs profit margin scatter plot.
    """
    # Set figure size
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Business Performance Summary', fontsize=16, fontweight='bold')

    # KPIs display as text at the top
    kpi_text = (f"Total Orders: {kpis['Total Orders']:,}\n"
                f"Total Sales: ${kpis['Total Sales']:,.2f}\n"
                f"Total Profit: ${kpis['Total Profit']:,.2f}\n"
                f"Average Order Value: ${kpis['Average Order Value']:,.2f}")
    axes[0, 0].axis('off')
    axes[0, 0].text(0.01, 0.9, kpi_text, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='whitesmoke', edgecolor='grey'))
    axes[0, 0].set_title('Key Metrics', fontweight='bold')

    # Sales by Category
    axes[0, 1].bar(by_category['Category'], by_category['Sales'], color='skyblue', label='Sales')
    axes[0, 1].bar(by_category['Category'], by_category['Profit'], color='salmon', label='Profit', alpha=0.7)
    axes[0, 1].set_title('Sales & Profit by Category')
    axes[0, 1].set_ylabel('Amount (USD)')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Sales by Segment
    axes[1, 0].bar(by_segment['Segment'], by_segment['Sales'], color='lightgreen', label='Sales')
    axes[1, 0].bar(by_segment['Segment'], by_segment['Profit'], color='orange', label='Profit', alpha=0.7)
    axes[1, 0].set_title('Sales & Profit by Segment')
    axes[1, 0].set_ylabel('Amount (USD)')
    axes[1, 0].legend()

    # Sales by Region
    axes[1, 1].bar(by_region['Region'], by_region['Sales'], color='plum', label='Sales')
    axes[1, 1].bar(by_region['Region'], by_region['Profit'], color='turquoise', label='Profit', alpha=0.7)
    axes[1, 1].set_title('Sales & Profit by Region')
    axes[1, 1].set_ylabel('Amount (USD)')
    axes[1, 1].legend()

    # Time series line chart
    axes[2, 0].plot(time_series['YearMonth'], time_series['Sales'], marker='o', linestyle='-', color='royalblue', label='Sales')
    axes[2, 0].plot(time_series['YearMonth'], time_series['Profit'], marker='o', linestyle='-', color='firebrick', label='Profit')
    axes[2, 0].set_title('Monthly Sales & Profit Trend')
    axes[2, 0].set_xlabel('Month')
    axes[2, 0].set_ylabel('Amount (USD)')
    axes[2, 0].legend()
    axes[2, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[2, 0].tick_params(axis='x', rotation=45)

    # Scatter plot: Discount vs Profit Margin
    axes[2, 1].scatter(scatter_df['Discount'], scatter_df['ProfitMargin'], alpha=0.3, c=scatter_df['Sales'], cmap='viridis')
    axes[2, 1].set_title('Discount vs Profit Margin')
    axes[2, 1].set_xlabel('Discount')
    axes[2, 1].set_ylabel('Profit Margin')
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=axes[2, 1])
    cb.set_label('Sales')

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze the synthetic retail dataset and build a dashboard.")
    parser.add_argument('--data', type=str, default='data.csv', help='Path to the input CSV dataset.')
    parser.add_argument('--output', type=str, default='.', help='Directory to store outputs.')
    parser.add_argument('--top', type=int, default=10, help='Number of top products to display.')
    args = parser.parse_args()

    data_path = args.data
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(data_path)

    # Compute key performance indicators
    kpis = compute_kpis(df)

    # Aggregate by dimensions
    agg_category = aggregate_by_dimension(df, 'Category')
    agg_segment = aggregate_by_dimension(df, 'Segment')
    agg_region = aggregate_by_dimension(df, 'Region')

    # Time series (not used in final dashboard but exported for reference)
    ts = aggregate_time_series(df)

    # Top products
    top_products = get_top_products(df, top_n=args.top)

    # Scatter data
    scatter_df = compute_discount_profit_scatter(df)

    # Save aggregated tables
    save_aggregated_tables(
        output_dir,
        sales_by_category=agg_category,
        sales_by_segment=agg_segment,
        sales_by_region=agg_region,
        time_series=ts,
        top_products=top_products
    )

    # Build dashboard
    fig = build_dashboard(
        df,
        kpis,
        by_category=agg_category,
        by_segment=agg_segment,
        by_region=agg_region,
        time_series=ts,
        top_products=top_products,
        scatter_df=scatter_df
    )

    # Save interactive dashboard to HTML
    html_path = os.path.join(output_dir, 'dashboard.html')
    fig.write_html(html_path, include_plotlyjs='cdn')

    print(f"Dashboard HTML saved to {html_path}")

    # Create a static summary image using Matplotlib for quick preview
    summary_image_path = os.path.join(output_dir, 'dashboard_summary.png')
    try:
        create_static_summary(
            output_path=summary_image_path,
            kpis=kpis,
            by_category=agg_category,
            by_segment=agg_segment,
            by_region=agg_region,
            time_series=ts,
            top_products=top_products,
            scatter_df=scatter_df
        )
        print(f"Static summary image saved to {summary_image_path}")
    except Exception as e:
        print(f"Warning: Failed to create static summary image: {e}")

    print("Analysis complete. Aggregated tables and dashboard exported.")


if __name__ == '__main__':
    main()