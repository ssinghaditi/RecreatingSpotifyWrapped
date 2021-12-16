# Aditi Singh
# ITP 216 (32081) - Fall 2021
# Final Project

# import all necessary packages
from flask import Flask, redirect, render_template, request, session, url_for, send_file
import io
from io import BytesIO
import os
import csv
import sqlite3 as sl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use("Agg")


app = Flask(__name__)

# Description: Gets the home page; 1st GET request
# Parameters: 0
# Returns: Template for home page
# 1st GET request
@app.route("/")
def home():
    return render_template('home.html', numminutes=str(round(totalnumminutes()[0]/60000)), songname=songnameplayed()[1],
                           artistname=songnameplayed()[0], songplayed=songnameplayed()[2])


# Description: Checks which radiobutton the client chose from user input; 1st POST request
# Parameters: 0
# Returns: Redirects to appropriate dynamic GET endpoint
@app.route("/viz", methods=["POST"])
def visualization():
    if request.method == "POST":
        inp = request.form["mins"]
    return redirect(url_for('rdirect', inp=inp))


# Description: Checks if the client chose the radio button to visualize artists or songs and gets the respective
# endpoint; 2nd (dynamic) GET request
# Parameters: 1 (inp - either song or artist)
# Returns: Redirects to appropriate endpoints
@app.route("/direct<inp>", methods=["POST"])
def rdirect(inp):
    if inp == "songs":
        # GET request
        return redirect(url_for('songViz'))
    else:
        # GET request
        return redirect(url_for('artistViz'))

# Description: Redirects to artist html page
# Parameters: 0
# Returns: Redirects to appropriate endpoints
@app.route("/directartists")
def artistViz():
    return render_template('artist.html')


# Description: Creates the visualization between top 10 artists and minutes streamed
# Parameters: 0
# Returns: png image of plot viz
@app.route("/plot1")
def plot1():
    data = extract10artists()
    artist = []
    playtime = []
    for x in data:
        artist.append(x[0])
        playtime.append(x[1]/60000)

    fig, ax1 = plt.subplots(1)
    ax1.bar(artist, playtime)
    ax1.set(title="Top 10 artists and number of minutes streamed in total")
    plt.xlabel("Artist name")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Minutes streamed")
    fig.tight_layout()

    img_bytes = BytesIO()
    fig.savefig(img_bytes)
    img_bytes.seek(0)
    return send_file(img_bytes, mimetype='image/png')


# Description: Redirects to song html page
# Parameters: 0
# Returns: Redirects to appropriate endpoints
@app.route("/directsongs")
def songViz():
    return render_template('song.html')


# Description: Creates the visualization between top 10 songs and minutes streamed
# Parameters: 0
# Returns: png image of plot viz
@app.route("/plot2")
def plot2():
    data = extract10songs()
    song = []
    playtime = []
    for x in data:
        song.append(x[0])
        playtime.append(x[1]/60000)

    fig, ax1 = plt.subplots(1)
    ax1.bar(song, playtime)
    ax1.set(title="Top 10 songs and number of minutes streamed in total")
    plt.xlabel("Song name")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Minutes streamed")
    fig.tight_layout()

    img_bytes = BytesIO()
    fig.savefig(img_bytes)
    img_bytes.seek(0)
    return send_file(img_bytes, mimetype='image/png')


# Description: Predicts an artist's popularity based on the minutes entered by the user (gotten through the post request
# ) and calls the function that deals with machine learning
# Parameters: 0
# Returns: template for the prediction html that shows artist's popularity based on minutes streamed of their music
# 2nd POST request
@app.route("/predict", methods=["POST", "GET"])
def prediction():
    if request.method == "POST":
        minute = request.form["min"]
    df = mutatedataset()
    pred = ml(df, float(minute))
    if pred == 1:
        pop = "popular"
    elif pred == 2:
        pop = "average"
    else:
        pop = "not popular"
    # GET request
    return render_template('prediction.html', min=minute, prediction=pop)


# Description: Given a month inputted by the user, finds common statistics like min, max, avg, sum of the minutes played
# of music that month
# Parameters: 0
# Returns: template for the scientific computation part
@app.route("/month", methods=["POST", "GET"])
def month():
    if request.method == "POST":
        month = request.form["month"]
    min, max, avg, sum = scientific_comp(month) # used pandas for these aggregations
    return render_template('month.html', month=month, min=round(min), max=round(max), avg=round(avg), sum=round(sum))


# Description: find the total ms played of music in the past year
# Parameters: 0
# Returns: result of the SQL query
def totalnumminutes():
    data = curs.execute("SELECT SUM(msPlayed) FROM streaming;")
    for record in data:
        return record


# Description: finds the artist of the song that was most played and also finds out how many times that particular
# song was played
# Parameters: 0
# Returns: result of the SQL query
def songnameplayed():
    data = curs.execute("SELECT artistName, trackName, count(*) as cnt FROM streaming WHERE msPlayed != 0 GROUP BY "
                        "trackName ORDER BY cnt desc LIMIT 1;")
    for record in data:
        return record


# Description: finds the top 10 artists based on the number of ms of songs played by them
# Parameters: 0
# Returns: result of the SQL query
def extract10artists():
    data = curs.execute("SELECT artistName, SUM(msPlayed) as total FROM streaming GROUP BY artistName ORDER BY total "
                        "desc LIMIT 10;")
    return data


# Description: find the top 10 songs played by the user based on ms played
# Parameters: 0
# Returns: result of the SQL query
def extract10songs():
    data = curs.execute("SELECT trackName, SUM(msPlayed) as total FROM streaming GROUP BY trackName ORDER BY total "
                        "desc LIMIT 10;")
    return data


# Description: mutates the dataset to create a new variable called artistPopularity that measures how popular a
# particular artists' music based on the ms of songs played by them
# not popular(3): between 0ms to the mean ms played
# average(2): between mean ms played to the average between mean ms played to max ms played
# popular(1): between the average between mean ms played and max ms played to max ms played (due to very high outliers)
# Parameters: 0
# Returns: dataframe after the SQL query and manipulation
# using pandas to manipulate data
def mutatedataset():
    data = curs.execute("SELECT artistName, SUM(msPlayed) as totalPlayed FROM streaming GROUP BY artistName")
    cols = [column[0] for column in data.description]
    df = pd.DataFrame.from_records(data=data.fetchall(), columns=cols)

    df = df.assign(artistPopularity=pd.cut(df['totalPlayed'],
                                           bins=[-1, round(df["totalPlayed"].mean()),
                                                 round((df["totalPlayed"].mean()+df["totalPlayed"].max())/2),
                                                 df["totalPlayed"].max()],
                                           labels=['3', '2', '1']))
    return df


# Description: deals with the machine learning aspect. predicts an artist's popularity based on the given ms played
# input data - used ms played as the feature and artist performance as the label
# Parameters: 2 (df - dataframe resulting from the mutatedataset function and test which is the input mins of songs
# played by the user)
# Returns: the prediction given the input by the user - integer form of whether the artist is going to be popular,
# average, or not popular
def ml(df, test):
    df["totalPlayed"] /= 60000
    df["totalPlayed"] = pd.to_numeric(df["totalPlayed"])
    df["artistPopularity"] = pd.to_numeric(df["artistPopularity"])
    data = df[["totalPlayed"]]
    target = df["artistPopularity"]
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train.values, y_train.values)
    pred = knn.predict(np.array([[test]]))
    return pred


# Description: deals with the action computation; use SQL to filter out the specific month given by the user and then
# uses pandas to find aggregate computations that gives insights into user's listening habits in a month
# Parameters: 1 (month - inputted by the user)
# Returns: 4: min, max, avg, sum of the ms/minutes played of music that specific month
def scientific_comp(month):
    data = curs.execute("SELECT * FROM streaming WHERE endTime like '"+month+"%';")
    cols = [column[0] for column in data.description]
    df = pd.DataFrame.from_records(data=data.fetchall(), columns=cols)
    return df['msPlayed'].min()/60000, df['msPlayed'].max()/60000, df['msPlayed'].mean()/60000, df['msPlayed'].sum()/60000


if __name__ == "__main__":
    conn = sl.connect("streaminghistory.db", check_same_thread=False)
    curs = conn.cursor()
    app.secret_key = os.urandom(12)
    app.run(debug=True)

    # had to comment the following code because used it to create the database. After 1 execution, the database is
    # created so no need to run this code again.
    """
    # create database
    conn = sl.connect("streaminghistory.db")
    curs = conn.cursor()

    # create new table inside database
    curs.execute("CREATE TABLE streaming('endTime','artistName' varchar(32),'trackName' varchar(32), 'msPlayed' int);")
    conn.commit()

    # insert data from the 4 csv into the database
    file1 = open("data_files/StreamingHistory0.csv")
    file2 = open("data_files/StreamingHistory1.csv")
    file3 = open("data_files/StreamingHistory2.csv")
    file4 = open("data_files/StreamingHistory3.csv")
    rows1 = csv.reader(file1)
    rows2 = csv.reader(file2)
    rows3 = csv.reader(file3)
    rows4 = csv.reader(file4)
    curs.executemany("INSERT INTO streaming VALUES (?, ?, ?, ?)", rows1)
    curs.executemany("INSERT INTO streaming VALUES (?, ?, ?, ?)", rows2)
    curs.executemany("INSERT INTO streaming VALUES (?, ?, ?, ?)", rows3)
    curs.executemany("INSERT INTO streaming VALUES (?, ?, ?, ?)", rows4)
    conn.commit()
    """
    conn.close()
