#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
import matplotlib.pyplot as plt, matplotlib.colors as clrs
from scipy.interpolate import interp1d
from datetime import date

FILE = 'data/_chat.txt'
COLORS = ['#d3d3d3', '#a9a9a9', '#588c7e', '#f2e394', '#f2ae72', '#d96459', '#8c4646']
DAYS = 0

members = []

class Member:
    """Represents one chat participator"""

    def __init__(self, name):
        self.name = name
        self.wc = 0  # Word count
        self.words = {}
        self.hours = [0 for _ in range(24)]

    def add_message(self, message, hour):
        self.hours[hour] += 1
        for word in message.split():
            word = word.lower()
            while len(word) > 1 and word[-1] in '-,."\'!?:)—':
                word = word[:-1]
            while len(word) > 1 and word[0] in ',."\'*(-—~':
                word = word[1:]
            self.words.setdefault(word, 0)
            self.words[word] += 1
            self.wc += 1


def date_diff_to_days(chat):
    """Calculates number of days that lie between the first and last message"""
    data = open(chat, 'r')
    lines = data.readlines()
    d0 = lines[0]
    d0 = date(2000+int(d0[6:8]), int(d0[3:5]), int(d0[:2]))
    d1 = lines[-1]
    d1 = date(2000+int(d1[6:8]), int(d1[3:5]), int(d1[:2]))
    return (d1 - d0).days


def process(chat):
    """Reads chat file and extracts data"""
    with open(chat, 'r') as data:
        for line in data:
            try:
                line = line[line.index(', ') + 2:]
                hour = int(line[:line.index(':')])
                line = line[line.index(': ') + 2:]
                name = line[:line.index(':')].decode('utf-8')  # no support for phone numbers
                line = line[line.index(': ') + 2:]
            except ValueError:
                pass
            else:
                if all(member.name != name for member in members):
                    members.append(Member(name))
                for member in members:
                    if member.name == name:
                        member.add_message(line, hour)
    global DAYS
    DAYS = date_diff_to_days(chat)


def plot_general():
    """Visualizes data concerning all the users"""
    global members

    # Set window title
    plt.figure().canvas.set_window_title('Whatsapp Analyzer')

    # Plot overall message count average per hour of the day
    x = np.linspace(0, 23, num=128, endpoint=True)
    y = [e / float(DAYS) for e in [sum([m.hours[i] for m in members]) for i in range(24)]]
    f = interp1d([i for i in range(24)], y, kind='cubic')
    plt.subplot(211)
    plt.plot(x, f(x), 'red')
    plt.xticks(np.arange(min(x), max(x)+1, 2.0))
    plt.grid()
    plt.xlabel('Hour of the Day')
    plt.ylabel('#Messages')
    plt.title('Average Messages per Hour')

    # Plot highest user specific message count average per hour of the day
    y, lbs = [], []
    for i in range(24):
        mx_l, mx_m = '', -1
        for m in members:
            if m.hours[i] > mx_m:
                mx_m = m.hours[i]
                mx_l = m.name.split()[0][:1] + '.' + m.name.split()[1][:1] + '.'
        y.append(mx_m / float(days))
        lbs.append(mx_l)
    plt.scatter(range(24), y, color='blue')
    for i in range(24):
        plt.annotate(lbs[i], (i, y[i]), xytext=(5, 5), textcoords='offset points') 
    plt.legend(['All Users Together', 'Most Active User in that Hour'], loc=2)

    # Show Plots
    plt.show()


def plot_user_spec():
    """Visualizes data concerning the single users"""
    global members

    # Set window title
    plt.figure().canvas.set_window_title('Whatsapp Analyzer')
    # Set colors
    colors = COLORS[:len(members)] if len(members) <= len(COLORS) else random.sample(clrs.cnames, len(members))

    # Total message count for each member as bar graph
    members = sorted(members, key=lambda m: sum(m.hours))
    plt.subplot(221)
    msgs = [sum(m.hours) for m in members]
    barlst = plt.barh(range(len(members)), msgs, align='center', height=0.4)
    # Set bar colors
    for i in range(len(barlst)):
        barlst[i].set_color(colors[i])
    plt.xlim([0, msgs[-1]*1.15])
    plt.yticks(range(len(members)), [m.name for m in members], size='small')
    # Annotate bars
    for i in range(len(members)):
        plt.text(msgs[i]+msgs[-1]*0.02, i, str(msgs[i]), ha='left', va='center')
    # Label plot and axes
    plt.title('Total Messages Sent during %d Days' % DAYS)

    # Total message count for each member as pie chart
    m_pie = plt.subplot(222)
    # Explode max
    explode = tuple([0 if i else 0.1 for i in range(len(members))])[::-1]
    plt.pie(msgs, explode=explode, labels=[' ' for m in members], colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Total Messages Sent as Share')
    # Configure legend
    handles, labels = m_pie.get_legend_handles_labels()
    plt.legend(handles[::-1], [m.name for m in members][::-1], loc='center', bbox_to_anchor=(0.95, 0.5))

    # Words per message for each member as bar graph
    plt.subplot(223)
    wc_avg = [m.wc / float(sum(m.hours)) for m in members]
    barlst = plt.barh(range(len(members)), wc_avg, align='center', height=0.4)
    # Set bar colors
    for i in range(len(barlst)):
        barlst[i].set_color(colors[i])
    plt.xlim([0, max(wc_avg)*1.15])
    plt.yticks(range(len(members)), [m.name for m in members], size='small')
    # Annotate bars
    for i in range(len(members)):
        plt.text(wc_avg[i]+max(wc_avg)*0.02, i, str(round(wc_avg[i], 3)), ha='left', va='center')
    plt.title('Average Words per Message')

    # Total word count for each member as pie chart
    w_pie = plt.subplot(224)
    # Explode max
    explode = tuple([0.1 if m.wc==max([n.wc for n in members]) else 0 for m in members])
    wc_total = [m.wc for m in members]
    plt.pie(wc_total, explode=explode, labels=[' ' for m in members], colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Total Words Written as Share')
    # Configure legend
    handles, labels = w_pie.get_legend_handles_labels()
    plt.legend(handles[::-1], [m.name for m in members][::-1], loc='center', bbox_to_anchor=(0.95, 0.5))

    # Show plots
    plt.show()


# Read file and process
process(FILE)
#plot_general()
plot_user_spec()

