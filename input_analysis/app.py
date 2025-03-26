#!/usr/bin/env python3
"""
Copyright (C) 2025.
Python script for BDD Dataset Analysis
Date: 26th March 2025
Authors:
- Gauresh Shirodkar
"""

# Standard imports
import os
import json
from collections import defaultdict

# Third party imports
import dash
import numpy as np
from dash import dcc, html
import plotly.express as px
from dash.dependencies import Input, Output


class BDDDashboard:
    def __init__(self, train_path, val_path):
        self.train_path = train_path
        self.val_path = val_path
        self.data_train = self.load_labels(self.train_path)

        self.data_val = self.load_labels(self.val_path)

        self.train_analysis = self.analyze_dataset(self.data_train)
        self.val_analysis = self.analyze_dataset(self.data_val)

        self.train_counts, self.train_sizes, self.train_positions = self.process_data(self.data_train)
        self.val_counts, self.val_sizes, self.val_positions = self.process_data(self.data_val)

    def load_labels(self, path):
        try:
            with open(path, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Error loading file at {path}")
            return []

    def analyze_dataset(self, labels):
        categories = defaultdict(int)
        weather_conditions = defaultdict(int)
        time_of_day = defaultdict(int)
        resolutions = []
        count_outlier = 0

        for entry in labels:
            for label in entry.get("labels", []):
                categories[label["category"]] += 1

            weather_conditions[entry.get("attributes", {}).get("weather", "unknown")] += 1
            time_of_day[entry.get("attributes", {}).get("timeofday", "unknown")] += 1

            for obj in entry['labels']:
                if obj['category'] in ['person', 'traffic light', 'truck', 'bus', 'bike', 'car', 'rider', 'traffic sign']:
                    width = obj.get("box2d", {}).get("x2", 0) - obj.get("box2d", {}).get("x1", 0)
                    height = obj.get("box2d", {}).get("y2", 0) - obj.get("box2d", {}).get("y1", 0)
                    if height and width:
                        resolutions.append((width, height))
                    if height < 20 and width < 20:
                        count_outlier += 1

        return categories, weather_conditions, time_of_day, resolutions

    def process_data(self, labels):
        object_counts = defaultdict(int)
        object_sizes = defaultdict(list)
        object_positions = defaultdict(list)

        for entry in labels:
            for label in entry.get("labels", []):
                obj_class = label["category"]
                object_counts[obj_class] += 1

                box2d = label.get("box2d", {})
                if box2d:
                    width = box2d["x2"] - box2d["x1"]
                    height = box2d["y2"] - box2d["y1"]
                    object_sizes[obj_class].append(width * height)
                    object_positions[obj_class].append(((box2d["x1"] + box2d["x2"]) / 2,
                                                        (box2d["y1"] + box2d["y2"]) / 2))
        return object_counts, object_sizes, object_positions

def main():
    TRAIN_LABELS_PATH = os.getenv("TRAIN_LABELS_PATH", "/data/train_data.json")
    VAL_LABELS_PATH = os.getenv("VAL_LABELS_PATH", "/data/val_data.json")

    dashboard = BDDDashboard(TRAIN_LABELS_PATH, VAL_LABELS_PATH)

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("BDD100K Dataset Analysis Dashboard"),
        dcc.Dropdown(
            id='dataset-dropdown',
            options=[
                {'label': 'Train Set', 'value': 'train'},
                {'label': 'Validation Set', 'value': 'val'}
            ],
            value='train',
            clearable=False
        ),
        dcc.Graph(id='category-distribution'),
        dcc.Graph(id='weather-distribution'),
        dcc.Graph(id='timeofday-distribution'),
        dcc.Graph(id='object-frequency'),
        dcc.Graph(id='object-size'),
        dcc.Graph(id='object-position')
    ])

    @app.callback(
        [Output('category-distribution', 'figure'),
         Output('weather-distribution', 'figure'),
         Output('timeofday-distribution', 'figure'),
         Output('object-frequency', 'figure'),
         Output('object-size', 'figure'),
         Output('object-position', 'figure')],
        [Input('dataset-dropdown', 'value')]
    )
    def update_graph(selected_dataset):
        analysis_data = dashboard.train_analysis if selected_dataset == 'train' else dashboard.val_analysis
        counts, sizes, positions = (dashboard.train_counts, dashboard.train_sizes, dashboard.train_positions) if selected_dataset == 'train' else (dashboard.val_counts, dashboard.val_sizes, dashboard.val_positions)

        categories, weather_conditions, time_of_day, resolutions = analysis_data

        category_fig = px.bar(x=list(categories.keys()), y=list(categories.values()),
                              labels={'x': 'Category', 'y': 'Count'},
                              title="Category Distribution")

        weather_fig = px.bar(x=list(weather_conditions.keys()), y=list(weather_conditions.values()),
                             labels={'x': 'Weather', 'y': 'Count'},
                             title="Weather Distribution")

        timeofday_fig = px.bar(x=list(time_of_day.keys()), y=list(time_of_day.values()),
                               labels={'x': 'Time of Day', 'y': 'Count'},
                               title="Time of Day Distribution")

        freq_fig = px.bar(x=list(counts.keys()), y=list(counts.values()),
                          labels={'x': 'Object Classes', 'y': 'Frequency'},
                          title="Object Frequency Distribution")

        size_data = {k: np.mean(v) for k, v in sizes.items()}
        size_fig = px.bar(x=list(size_data.keys()), y=list(size_data.values()),
                          labels={'x': 'Object Classes', 'y': 'Average Size'},
                          title="Average Object Size")

        position_data = []
        for obj_class, pos_list in positions.items():
            for pos in pos_list:
                position_data.append({'Class': obj_class, 'X': pos[0], 'Y': pos[1]})
        position_fig = px.scatter(position_data, x='X', y='Y', color='Class',
                                  title="Object Position Distribution")

        return category_fig, weather_fig, timeofday_fig, freq_fig, size_fig, position_fig

    # app.run(debug=True, host='0.0.0.0', port=8050)
    app.run(debug=True, port=8050)
    # app.run(debug=True)

if __name__ == '__main__':
    main()


"""
docker run --rm -p 8050:8050 -v "C:\Personal\Bosch_assignment\:/data" -e TRAIN_LABELS_PATH=/data/assignment_data_bdd_files/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json -e VAL_LABELS_PATH=/data/assignment_data_bdd_files/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json python-data-container

"""