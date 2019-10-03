#!/usr/bin/python3

import pandas as pd
import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from headers import *
from flask import render_template


#Function that reades the header and saves it into a list -> header
def show_headers(in_file):
    header = []
    df = pd.read_csv(in_file)

    for i in list(df):
        header.append(i)

    for i in header:
        print i
    return header
