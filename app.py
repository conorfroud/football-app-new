import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import font_manager
from highlight_text import fig_text
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from scipy import stats
import plotly.graph_objs as go
from math import pi
from mplsoccer import Pitch
from mplsoccer import PyPizza
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from mplsoccer.pitch import Pitch, VerticalPitch
import io
import base64
import requests
from io import BytesIO

st.set_page_config(layout="wide")

pd.set_option("display.width", None)  # None means no width limit
       
def team_scatter_plot(df4):
    # Create three columns layout
    col1, col2, col3 = st.columns([1, 5, 1])

    with col2:
        def highlight_color(row):
            if row['team_name'] == 'Stoke City':
                return '#FF8080'  # light red
            elif row['Promoted?'] == 'Yes':
                return '#90EE90'  # light green
            elif row['Relegated?'] == 'Yes':
                return '#0066cc'  # light blue
            else:
                return 'grey'

        # Adjust opacity based on season
        def adjust_opacity(row):
            if row['season_name'] == '2024/2025':
                return 1.0  # fully opaque
            else:
                return 0.55  # more transparent

        # Filter dataframe for season '2024/2025'
        label_df = df4[df4['season_name'] == '2024/2025']

        # Function to add mean lines to a figure
        def add_mean_lines(fig, x_mean, y_mean, x_col, y_col):
            fig.add_shape(
                type='line',
                x0=x_mean,
                x1=x_mean,
                y0=df4[y_col].min(),
                y1=df4[y_col].max(),
                line=dict(dash='dot', color='black')
            )
            fig.add_shape(
                type='line',
                x0=df4[x_col].min(),
                x1=df4[x_col].max(),
                y0=y_mean,
                y1=y_mean,
                line=dict(dash='dot', color='black')
            )
            return fig

        # Create the first scatter plot using Plotly with the entire data
        x_mean = df4['xG'].mean()
        y_mean = df4['xG Conceded'].mean()
        fig1 = px.scatter(df4, x='xG', y='xG Conceded',
                          hover_data={'team_name': True, 'season_name': True, 'xG': True, 'xG Conceded': True},
                          trendline="ols")

        # Customize the marker color, size, and opacity
        fig1.update_traces(marker=dict(size=12,
                                       color=df4.apply(highlight_color, axis=1),
                                       opacity=df4.apply(adjust_opacity, axis=1)))

        # Access the trendline and customize its appearance
        fig1.data[-1].update(line=dict(color='black', dash='dot'))

        # Set the plot size, title, and reverse the y-axis for 'xG Conceded'
        fig1.update_layout(
            yaxis=dict(autorange='reversed'),
            width=800,
            height=600,
            title={
                'text': "xG Performance",
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(family="Roboto", size=20, color='black')
            }
        )

        # Add mean lines
        fig1 = add_mean_lines(fig1, x_mean, y_mean, 'xG', 'xG Conceded')

        # Label teams only from '2024/2025' season
        fig1.add_trace(
            go.Scatter(
                text=label_df['team_name'],
                x=label_df['xG'],
                y=label_df['xG Conceded'],
                mode='text',
                showlegend=False,
                textposition='top center'
            )
        )

        # Display the first plot in Streamlit
        st.plotly_chart(fig1)

        # Second scatter plot
        x_mean = df4['Non-Penalty Goals Scored'].mean()
        y_mean = df4['Non-Penalty Goals Conceded'].mean()
        fig2 = px.scatter(df4, x='Non-Penalty Goals Scored', y='Non-Penalty Goals Conceded',
                          hover_data={'team_name': True, 'season_name': True, 'Non-Penalty Goals Scored': True, 'Non-Penalty Goals Conceded': True}, trendline="ols")

        # Customize the marker color, size, and opacity
        fig2.update_traces(marker=dict(size=12,
                                       color=df4.apply(highlight_color, axis=1),
                                       opacity=df4.apply(adjust_opacity, axis=1)))

        # Access the trendline and customize its appearance
        fig2.data[-1].update(line=dict(color='black', dash='dot'))

        # Set the plot size and title
        fig2.update_layout(
            width=800,
            height=600,
            title={
                'text': "Goals Performance",
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(family="Roboto", size=20, color='black')
            },
            yaxis=dict(autorange="reversed")
        )

        # Add mean lines
        fig2 = add_mean_lines(fig2, x_mean, y_mean, 'Non-Penalty Goals Scored', 'Non-Penalty Goals Conceded')

        # Label teams only from '2023/2024' season
        fig2.add_trace(
            go.Scatter(
                text=label_df['team_name'],
                x=label_df['Non-Penalty Goals Scored'],
                y=label_df['Non-Penalty Goals Conceded'],
                mode='text',
                showlegend=False,
                textposition='top center'
            )
        )

        # Display the second plot in Streamlit
        st.plotly_chart(fig2)

        # Third scatter plot
        fig3 = px.scatter(df4, x='xG', y='Non-Penalty Goals Scored',
                          hover_data={'team_name': True, 'season_name': True, 'xG': True, 'Non-Penalty Goals Scored': True})

        # Customize the marker color, size, and opacity
        fig3.update_traces(marker=dict(size=12,
                                       color=df4.apply(highlight_color, axis=1),
                                       opacity=df4.apply(adjust_opacity, axis=1)))

        # Set the plot size and title
        fig3.update_layout(
            width=800,
            height=600,
            title={
                'text': "Attacking Over/Under Performance",
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(family="Roboto", size=20, color='black')
            }
        )

        # Add a y=x line
        fig3.add_trace(
            go.Scatter(
                x=[df4['xG'].min(), df4['xG'].max()],
                y=[df4['xG'].min(), df4['xG'].max()],
                mode='lines',
                line=dict(color='black', dash='dash'),
                showlegend=False
            )
        )

        # Label teams only from '2023/2024' season
        fig3.add_trace(
            go.Scatter(
                text=label_df['team_name'],
                x=label_df['xG'],
                y=label_df['Non-Penalty Goals Scored'],
                mode='text',
                showlegend=False,
                textposition='top center'
            )
        )

        # Display the third plot in Streamlit
        st.plotly_chart(fig3)

        # Fourth scatter plot
        fig4 = px.scatter(df4, x='xG Conceded', y='Non-Penalty Goals Conceded',
                          hover_data={'team_name': True, 'season_name': True, 'xG Conceded': True, 'Non-Penalty Goals Conceded': True})

        # Customize the marker color, size, and opacity
        fig4.update_traces(marker=dict(size=12,
                                       color=df4.apply(highlight_color, axis=1),
                                       opacity=df4.apply(adjust_opacity, axis=1)))


        # Set the plot size and title
        fig4.update_layout(
            width=800,
            height=600,
            title={
                'text': "Defensive Over/Under Performance",
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(family="Roboto", size=20, color='black')
            },
            yaxis=dict(autorange="reversed"),
            xaxis=dict(autorange="reversed")
        )

        # Add a y=x line
        fig4.add_trace(
            go.Scatter(
                x=[df4['xG Conceded'].min(), df4['xG Conceded'].max()],
                y=[df4['xG Conceded'].min(), df4['xG Conceded'].max()],
                mode='lines',
                line=dict(color='black', dash='dash'),
                showlegend=False
            )
        )

        # Label teams only from '2023/2024' season
        fig4.add_trace(
            go.Scatter(
                text=label_df['team_name'],
                x=label_df['xG Conceded'],
                y=label_df['Non-Penalty Goals Conceded'],
                mode='text',
                showlegend=False,
                textposition='top center'
            )
        )

        # Display the fourth plot in Streamlit
        st.plotly_chart(fig4)

        # Fifth scatter plot (same x and y, but invert axes)
        x_mean = df4['Passes Per Possession'].mean()
        y_mean = df4['Pace Towards Goal'].mean()
        fig5 = px.scatter(df4, x='Passes Per Possession', y='Pace Towards Goal',
                          hover_data={'team_name': True, 'season_name': True, 'Passes Per Possession': True, 'Pace Towards Goal': True}, trendline="ols")

        # Customize the marker color, size, and opacity
        fig5.update_traces(marker=dict(size=12,
                                       color=df4.apply(highlight_color, axis=1),
                                       opacity=df4.apply(adjust_opacity, axis=1)))

        # Set the plot size and title
        fig5.update_layout(
            width=800,
            height=600,
            title={
                'text': "Build-Up Style",
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(family="Roboto", size=20, color='black')
            },
            yaxis=dict(autorange="reversed")
        )

        # Add mean lines
        fig5 = add_mean_lines(fig5, x_mean, y_mean, 'Passes Per Possession', 'Pace Towards Goal')

        # Access the trendline and customize its appearance
        fig5.data[-1].update(line=dict(color='black', dash='dot'))

        # Label teams only from '2023/2024' season
        fig5.add_trace(
            go.Scatter(
                text=label_df['team_name'],
                x=label_df['Passes Per Possession'],
                y=label_df['Pace Towards Goal'],
                mode='text',
                showlegend=False,
                textposition='top center'
            )
        )

        # Display the fifth plot in Streamlit
        st.plotly_chart(fig5)

        # Sixth scatter plot
        x_mean = df4['Passes Per Defensive Action'].mean()
        y_mean = df4['Defensive Distance'].mean()
        fig6 = px.scatter(df4, x='Passes Per Defensive Action', y='Defensive Distance',
                          hover_data={'team_name': True, 'season_name': True, 'Passes Per Defensive Action': True, 'Defensive Distance': True}, trendline="ols")

        # Customize the marker color, size, and opacity
        fig6.update_traces(marker=dict(size=12,
                                       color=df4.apply(highlight_color, axis=1),
                                       opacity=df4.apply(adjust_opacity, axis=1)))


        # Set the plot size and title
        fig6.update_layout(
            width=800,
            height=600,
            title={
                'text': "Pressing",
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(family="Roboto", size=20, color='black')
            },
            xaxis=dict(autorange="reversed")
        )

        # Add mean lines
        fig6 = add_mean_lines(fig6, x_mean, y_mean, 'Passes Per Defensive Action', 'Defensive Distance')

        # Access the trendline and customize its appearance
        fig6.data[-1].update(line=dict(color='black', dash='dot'))

        # Label teams only from '2023/2024' season
        fig6.add_trace(
            go.Scatter(
                text=label_df['team_name'],
                x=label_df['Passes Per Defensive Action'],
                y=label_df['Defensive Distance'],
                mode='text',
                showlegend=False,
                textposition='top center'
            )
        )

        # Display the sixth plot in Streamlit
        st.plotly_chart(fig6)

def all_team_scatter_plot(df4):
    # Sidebar filter for 'competition_name' (single select)
    competitions = df4['competition_name'].unique()
    selected_competition = st.sidebar.selectbox(
        'Select Competition', 
        options=competitions, 
        index=0  # Default to the first competition
    )

    # Filter the dataframe based on selected competition
    df_filtered = df4[df4['competition_name'] == selected_competition]

    # Create three columns layout
    col1, col2, col3 = st.columns([1, 5, 1])

    with col2:
        # Adjust opacity based on season
        def adjust_opacity(row):
            if row['season_name'] == '2024/2025':
                return 1.0  # fully opaque
            else:
                return 0.55  # more transparent

        # Filter dataframe for season '2024/2025'
        label_df = df_filtered[df_filtered['season_name'] == '2024/2025']

        # Function to add mean lines to a figure
        def add_mean_lines(fig, x_mean, y_mean, x_col, y_col):
            fig.add_shape(
                type='line',
                x0=x_mean,
                x1=x_mean,
                y0=df_filtered[y_col].min(),
                y1=df_filtered[y_col].max(),
                line=dict(dash='dot', color='black')
            )
            fig.add_shape(
                type='line',
                x0=df_filtered[x_col].min(),
                x1=df_filtered[x_col].max(),
                y0=y_mean,
                y1=y_mean,
                line=dict(dash='dot', color='black')
            )
            return fig

        ### First Scatter Plot: xG vs. xG Conceded ###
        x_mean_xg = df_filtered['xG'].mean()
        y_mean_xg_conceded = df_filtered['xG Conceded'].mean()
        fig1 = px.scatter(df_filtered, x='xG', y='xG Conceded',
                          hover_data={'team_name': True, 'season_name': True, 'xG': True, 'xG Conceded': True},
                          trendline="ols")

        # Customize the marker size, opacity, and color
        fig1.update_traces(marker=dict(size=12,
                                       color='grey',  # Set point color to grey
                                       opacity=df_filtered.apply(adjust_opacity, axis=1)))

        # Access the trendline and customize its appearance
        fig1.data[-1].update(line=dict(color='black', dash='dot'))

        # Set the plot size, title, and reverse the y-axis for 'xG Conceded'
        fig1.update_layout(
            yaxis=dict(autorange='reversed'),
            width=800,
            height=600,
            title={
                'text': "xG Performance",
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(family="Roboto", size=20, color='black')
            }
        )

        # Add mean lines
        fig1 = add_mean_lines(fig1, x_mean_xg, y_mean_xg_conceded, 'xG', 'xG Conceded')

        # Label teams only from '2024/2025' season
        fig1.add_trace(
            go.Scatter(
                text=label_df['team_name'],
                x=label_df['xG'],
                y=label_df['xG Conceded'],
                mode='text',
                showlegend=False,
                textposition='top center'
            )
        )

        # Display the first plot in Streamlit
        st.plotly_chart(fig1)

        ### Second Scatter Plot: Passes Per Possession vs. Pace Towards Goal ###
        x_mean_ppp = df_filtered['Passes Per Possession'].mean()
        y_mean_pace = df_filtered['Pace Towards Goal'].mean()
        fig2 = px.scatter(df_filtered, x='Passes Per Possession', y='Pace Towards Goal',
                          hover_data={'team_name': True, 'season_name': True, 'Passes Per Possession': True, 'Pace Towards Goal': True},
                          trendline="ols")

        # Customize the marker size, opacity, and color
        fig2.update_traces(marker=dict(size=12,
                                       color='grey',  # Set point color to grey
                                       opacity=df_filtered.apply(adjust_opacity, axis=1)))

        # Access the trendline and customize its appearance
        fig2.data[-1].update(line=dict(color='black', dash='dot'))

        # Set the plot size, title, and flip the y-axis
        fig2.update_layout(
            yaxis=dict(autorange='reversed'),  # Flip the Y-axis
            width=800,
            height=600,
            title={
                'text': "Build-Up Style",
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(family="Roboto", size=20, color='black')
            }
        )

        # Add mean lines
        fig2 = add_mean_lines(fig2, x_mean_ppp, y_mean_pace, 'Passes Per Possession', 'Pace Towards Goal')

        # Label teams only from '2024/2025' season
        fig2.add_trace(
            go.Scatter(
                text=label_df['team_name'],
                x=label_df['Passes Per Possession'],
                y=label_df['Pace Towards Goal'],
                mode='text',
                showlegend=False,
                textposition='top center'
            )
        )

        # Display the second plot in Streamlit
        st.plotly_chart(fig2)

        ### Third Scatter Plot: Passes Per Defensive Action vs. Defensive Distance ###
        x_mean_ppda = df_filtered['Passes Per Defensive Action'].mean()
        y_mean_dd = df_filtered['Defensive Distance'].mean()
        fig3 = px.scatter(df_filtered, x='Passes Per Defensive Action', y='Defensive Distance',
                          hover_data={'team_name': True, 'season_name': True, 'Passes Per Defensive Action': True, 'Defensive Distance': True},
                          trendline="ols")

        # Customize the marker size, opacity, and color
        fig3.update_traces(marker=dict(size=12,
                                       color='grey',  # Set point color to grey
                                       opacity=df_filtered.apply(adjust_opacity, axis=1)))

        # Access the trendline and customize its appearance
        fig3.data[-1].update(line=dict(color='black', dash='dot'))

        # Set the plot size, title, and flip both the x-axis and y-axis
        fig3.update_layout(
            xaxis=dict(autorange='reversed'),  # Flip the X-axis
            yaxis=dict(autorange='reversed'),  # Flip the Y-axis
            width=800,
            height=600,
            title={
                'text': "Pressing",
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(family="Roboto", size=20, color='black')
            }
        )

        # Add mean lines
        fig3 = add_mean_lines(fig3, x_mean_ppda, y_mean_dd, 'Passes Per Defensive Action', 'Defensive Distance')

        # Label teams only from '2024/2025' season
        fig3.add_trace(
            go.Scatter(
                text=label_df['team_name'],
                x=label_df['Passes Per Defensive Action'],
                y=label_df['Defensive Distance'],
                mode='text',
                showlegend=False,
                textposition='top center'
            )
        )

        # Display the third plot in Streamlit
        st.plotly_chart(fig3)

def team_rolling_averages_new(data1):

    window = 5  # Define your rolling window size
    team = 'Stoke City'  # Replace with your team name

    # Define thresholds for each metric
    thresholds = {
        'xG For': {'green_threshold': 1.15, 'orange_threshold': 1.05},
        'xG Per Shot For': {'green_threshold': 0.095, 'orange_threshold': 0.085},
        'xG Against': {'green_threshold': 1.0, 'orange_threshold': 1.12},
        'Shots For': {'green_threshold': 12.2, 'orange_threshold': 11},
        'Clear Shots For': {'green_threshold': 1.9, 'orange_threshold': 1.7},
        'xG Per Shot Against': {'green_threshold': 0.09, 'orange_threshold': 0.1},
        'Shots Against': {'green_threshold': 11, 'orange_threshold': 12.5},
        'Clear Shots Against': {'green_threshold': 1.7, 'orange_threshold': 1.9},
        'Deep Progressions For': {'green_threshold': 44, 'orange_threshold': 40},  # In Possession Metrics
        'Deep Completions For': {'green_threshold': 4.6, 'orange_threshold': 4.1},  # In Possession Metrics
        'Pass OBV For': {'green_threshold': 0.9, 'orange_threshold': 0.8},  # In Possession Metrics
        'Box Cross %': {'green_threshold': 30, 'orange_threshold': 35},  # In Possession Metrics
        'Deep Progressions Against': {'green_threshold': 36, 'orange_threshold': 40},  # Out of Possession Metrics
        '% of Pressures Opp Half': {'green_threshold': 45, 'orange_threshold': 40},  # Out of Possession Metrics (percentages)
        'Defensive Distance': {'green_threshold': 45, 'orange_threshold': 40},  # Out of Possession Metrics
        'High Press Shots Against': {'green_threshold': 2, 'orange_threshold': 2.4}  # Out of Possession Metrics
    }

    # Function to create the visualization
    def create_visualization(df, metric, team, window, green_threshold=1.2, orange_threshold=1.05, flip_colors=False):
        rolling = df[metric].rolling(window).mean()

        fig, ax = plt.subplots(figsize=(12, 6))  # Already consistent with Plotly
        fig.set_facecolor('White')
        ax.patch.set_facecolor('White')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('#ccc8c8')
        ax.spines['bottom'].set_color('#ccc8c8')

        x_pos = np.arange(len(df))

        ax.bar(x_pos, df[metric], color='black', alpha=0.75)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['Opponent'], rotation=90, fontsize=12, fontname="Roboto", color='black')  # Font standardization
        ax.plot(rolling, lw=3, color='red', markersize=5, zorder=10, label=f"{window} match rolling average")
        ax.grid(ls='dotted', lw=0.5, color='Black', zorder=1, alpha=0.4)

        ax.set_xlabel('Games', fontsize=12, fontname="Roboto", color='Black')
        ax.set_ylabel(metric, fontsize=12, fontname="Roboto", color='Black')

        if flip_colors:
            ax.axhspan(green_threshold, df[metric].max(), facecolor='red', alpha=0.1)   # Red on top
            ax.axhspan(orange_threshold, green_threshold, facecolor='orange', alpha=0.1)  # Orange in middle
            ax.axhspan(0, orange_threshold, facecolor='green', alpha=0.1)  # Green on bottom
        else:
            ax.axhspan(green_threshold, df[metric].max(), facecolor='green', alpha=0.1)  # Green on top
            ax.axhspan(orange_threshold, green_threshold, facecolor='orange', alpha=0.1)  # Orange in middle
            ax.axhspan(0, orange_threshold, facecolor='red', alpha=0.1)  # Red on bottom

        fig.suptitle(f"{team} {metric} | Trendline", color='Black', family="Roboto", fontsize=18, fontweight="bold", x=0.52, y=0.96)

        return fig

    # Sidebar for metric selection
    st.sidebar.title('Select Metric Group')
    metric_group = st.sidebar.selectbox(
        'Which group of metrics would you like to view?',
        ('Attacking Metrics', 'Defensive Metrics', 'In Possession Metrics', 'Out of Possession Metrics')  # Added 'Out of Possession Metrics'
    )

    # Create three columns layout
    col1, col2, col3 = st.columns([1, 5, 1])

    with col2:
        # Plot Attacking Metrics
        if metric_group == 'Attacking Metrics':
            
            fig_xg_for = create_visualization(data1, 'xG For', team, window, **thresholds['xG For'])
            st.pyplot(fig_xg_for)

            fig_xg_per_shot_for = create_visualization(data1, 'xG Per Shot For', team, window, **thresholds['xG Per Shot For'])
            st.pyplot(fig_xg_per_shot_for)

            fig_shots_for = create_visualization(data1, 'Shots For', team, window, **thresholds['Shots For'])
            st.pyplot(fig_shots_for)

            fig_clear_shots_for = create_visualization(data1, 'Clear Shots For', team, window, **thresholds['Clear Shots For'])
            st.pyplot(fig_clear_shots_for)

        # Plot Defensive Metrics
        elif metric_group == 'Defensive Metrics':
            
            fig_xg_against = create_visualization(data1, 'xG Against', team, window, **thresholds['xG Against'], flip_colors=True)
            st.pyplot(fig_xg_against)

            fig_xg_per_shot_against = create_visualization(data1, 'xG Per Shot Against', team, window, **thresholds['xG Per Shot Against'], flip_colors=True)
            st.pyplot(fig_xg_per_shot_against)

            fig_shots_against = create_visualization(data1, 'Shots Against', team, window, **thresholds['Shots Against'], flip_colors=True)
            st.pyplot(fig_shots_against)

            fig_clear_shots_against = create_visualization(data1, 'Clear Shots Against', team, window, **thresholds['Clear Shots Against'], flip_colors=True)
            st.pyplot(fig_clear_shots_against)

        # Plot In Possession Metrics
        elif metric_group == 'In Possession Metrics':  # New section for In Possession Metrics

            fig_deep_progressions_for = create_visualization(data1, 'Deep Progressions For', team, window, **thresholds['Deep Progressions For'])
            st.pyplot(fig_deep_progressions_for)

            fig_deep_completions_for = create_visualization(data1, 'Deep Completions For', team, window, **thresholds['Deep Completions For'])
            st.pyplot(fig_deep_completions_for)

            fig_pass_obv_for = create_visualization(data1, 'Pass OBV For', team, window, **thresholds['Pass OBV For'])
            st.pyplot(fig_pass_obv_for)

            fig_box_cross_pct = create_visualization(data1, 'Box Cross %', team, window, **thresholds['Box Cross %'], flip_colors=True)
            st.pyplot(fig_box_cross_pct)

        # Plot Out of Possession Metrics
        elif metric_group == 'Out of Possession Metrics':  # New section for Out of Possession Metrics

            fig_deep_progressions_against = create_visualization(data1, 'Deep Progressions Against', team, window, **thresholds['Deep Progressions Against'], flip_colors=True)
            st.pyplot(fig_deep_progressions_against)

            fig_pressures_opp_half = create_visualization(data1, '% of Pressures Opp Half', team, window, **thresholds['% of Pressures Opp Half'], flip_colors=False)
            st.pyplot(fig_pressures_opp_half)

            fig_defensive_distance = create_visualization(data1, 'Defensive Distance', team, window, **thresholds['Defensive Distance'], flip_colors=False)
            st.pyplot(fig_defensive_distance)

            fig_high_press_shots_against = create_visualization(data1, 'High Press Shots Against', team, window, **thresholds['High Press Shots Against'], flip_colors=True)
            st.pyplot(fig_high_press_shots_against)
    
# Load the DataFrame
df = pd.read_csv("belgiumdata.csv")
df2 = pd.read_csv("championshipscores.csv")
df3 = pd.read_csv("nonpriorityleaguesdata.csv")
df4 = pd.read_csv("teamseasondata.csv")
df5 = pd.read_csv("leaguesteamseasondata.csv")
data = pd.read_csv("seasonmatchdata2024.csv")
data1 = pd.read_csv("Stoke City Performance Data - Sheet1.csv")

# Create the navigation menu in the sidebar
selected_tab = st.sidebar.radio("Navigation", ["SCFC Team Data", "Rolling Average Data", "Team Data"])

# Based on the selected tab, display the corresponding content
if selected_tab == "Stoke Score - Wyscout":
    stoke_score_wyscout(df3)
if selected_tab == "Confidence Scores":
    display_data()
if selected_tab == "Player Profile":
    streamlit_interface(df2)
if selected_tab == "Report Search":
    searchable_reports()
if selected_tab == "Shortlist XI":
    shortlist_eleven()
if selected_tab == "Player Database":
    scouting_data()
if selected_tab == "SCFC Team Data":
    team_scatter_plot(df4)
if selected_tab == "Rolling Average Data New":
    team_rolling_averages(data)
if selected_tab == "Rolling Average Data":
    team_rolling_averages_new(data1)
if selected_tab == "Team Data":
    all_team_scatter_plot(df5)
elif selected_tab == "Multi Player Comparison Tab":
    comparison_tab(df)
