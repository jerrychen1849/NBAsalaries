# Jerry Chen, jchen710@usc.edu
# ITP 216, Spring 2023
# Section: 31883
# Final Project
# Description: This final project allows a user to graph average NBA salaries or projected NBA salaries until 2023

from flask import Flask, render_template, request, session, url_for, redirect
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from io import BytesIO
import base64

# Initialize our Flask app
app = Flask(__name__)


@app.route('/')
def home():
    """
    Allows client to choose average salary or projected salary
    Reads our CSV file and allows us to select an NBA team from drop down menu

    :return: renders home.html template with tram selection and option to choose avg/projected salary
    """
    # Read CSV file and convert to dataframe
    df = pd.read_csv('player_data.csv')
    # Remove all null values
    df['team'] = df['team'].fillna('Unknown')
    # Sort out the unique teams in our dataframe
    teams = sorted(df.loc[df['team'] != 'Unknown', 'team'].unique())

    # Display the two options for our model: one for past salaries and one for projected salaries
    options = {
        "past": "Average salary on team (1985-2018)",
        "projected": "Projected average salary"
    }
    # Render home.html template and teams is the drop-down menu with a list of all unique teams
    return render_template("home.html",
                           teams=teams,
                           options=options)


@app.route('/client', methods=["POST"])
def select_team_data():
    """
    This routes us to the correct function based on average/projected salary

    :return: redirects to home if user not logged in or redirects to average/projection function
    """
    # Set team session to choice from drop down menu
    session['team'] = request.form['team']
    # If no team selected, then redirect to home
    if 'team' not in session or session["team"] == "":
        return redirect(url_for("home"))
    # If option 1 selected, redirect to past data function
    if request.form['data_type'] == "past":
        return redirect(url_for("past_data", team=session["team"]))
    # If option 2 selected, redirect to projected data function
    elif request.form['data_type'] == "projected":
        return redirect(url_for("projected_data", team=session["team"]))
    # Otherwise, redirect to home
    else:
        return redirect(url_for("home"))


@app.route('/past_data')
def past_data():
    """
    Filters our dataframe to include only salary data for the selected team

    :return: past_data.html visualization function with team selected
    """
    # If no team selected, then redirect to home
    if 'team' not in session or session["team"] == "":
        return redirect(url_for("home"))
    # Set variable for team
    team = session['team']
    # Read CSV file
    df = pd.read_csv('player_data.csv')
    # Initialize variable for team column
    df = df[df['team'] == team]
    # Arrange dataframe by season start (year) and salary
    df = df[['season_start', 'salary']]
    # Group our dataframe by unique years
    df = df.groupby('season_start').mean()

    # Store the dataframe into the session as a JSON file
    session['df'] = df.to_json()
    
    # Return function to plot past data with team parameter being the team session
    return redirect(url_for("plot_past_data",
                            team=session["team"]))


@app.route('/past_data/<team>')
def plot_past_data(team):
    """
    Plots the data from our filtered dataframe into matplotlib visualization
    :param team:
    :return:
    """
    # If no team selected, then redirect to home
    if 'team' not in session or session["team"] == "":
        return redirect(url_for("home"))

    # Retrieve the JSON file from the past_data() function
    df = pd.read_json(session['df'])
    # Create our subplot
    fig, ax = plt.subplots()
    # Plot our index of years against salary
    ax.plot(df.index, df['salary'])
    # Set labels for x and y-axis
    ax.set_xlabel('Season')
    ax.set_ylabel('Average Salary')
    # Create title
    ax.set_title(team + ' Average Salary by Season')

    # Projecting our plot onto our flask webpage
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Render our past_data.html webpage with plot_url being our Matplotlib projection
    return render_template('past_data.html',
                           team=session["team"],
                           plot_url=plot_url)


@app.route('/projected_data')
def projected_data():
    """
    Reads dataframe and uses linear regression model to predict average salaries until 2030

    :return: Plot function for projection with years and salaries calculated
    """
    # If no team selected, then redirect to home
    if 'team' not in session or session["team"] == "":
        return redirect(url_for("home"))
    # Read CSV file and create dataframe
    df = pd.read_csv('player_data.csv')

    team = session['team']
    # Created indices for our unique teams
    df['futures'] = (df['team'] == team).astype(int)
    # Our x-axis is our season start based on the team
    x = df[['season_start', 'futures']]
    # Y-axis is salary
    y = df['salary']
    # Initialize our Linear Regression model
    model = LinearRegression()
    # Fit our model with our linear regression
    model.fit(x, y)

    # Create a Numpy array of years
    season_start_years = np.arange(2018, 2031)
    # Convert Numpy array to list and assign session variable for years as a list
    session['years'] = season_start_years.tolist()

    # Create a dataframe of future data from our regression with season on x-axis and salaries on y-xis
    future_data = pd.DataFrame({
        'season_start': season_start_years,
        'futures': np.where(season_start_years >= 2020, 1, 0)
    })

    # Futures salaries is a Numpy array of predicted values from linear regression model
    future_salaries = model.predict(future_data[['season_start', 'futures']])
    # Convert Numpy array to list and assign session variable to salaries
    session['salaries'] = future_salaries.tolist()

    # Return projection plot function with team parameter
    return redirect(url_for("plot_projected_data",
                            team=session['team']))


@app.route('/projected_data/<team>')
def plot_projected_data(team):
    """
    Plots projected data after given team information
    :param team: the team selected from our home menu, or the session team
    :return: A matplotlib projection of projected average salaries for NBA teams
    """
    # If no team selected, then redirect to home
    if 'team' not in session or session["team"] == "":
        return redirect(url_for("home"))
    # Set variables for session years and salaries
    years = session['years']
    salaries = session['salaries']

    # Create subplot for our regression
    fig, ax = plt.subplots()
    # Plot future seasons vs. predicted salaries
    ax.plot(years, salaries )
    # Set x and y-axis labels
    ax.set_xlabel('Season')
    ax.set_ylabel('Average Salary')
    # Set plot title
    ax.set_title(f'Predicted Average Salary for ' + team + ' Players (2019-2030)')

    # Projecting our plot onto our flask webpage (same process as for past_data)
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Render our projected data HTML page
    return render_template('projected_data.html',
                           team=session["team"],
                           plot_url=plot_url)

# Run our Flask webpage
if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True)
