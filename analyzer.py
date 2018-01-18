#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
import matplotlib.pyplot as plt, matplotlib.colors as clrs
from scipy.interpolate import interp1d
from datetime import date

FILE = 'data/_chat.txt'
COLORS = ['#d3d3d3', '#a9a9a9', '#588c7e', '#f2e394', '#f2ae72', '#d96459', '#8c4646']


class Member:
    """Represents one chat participator"""

    def __init__(self, name):
        """Initializes object and sets variables to default values"""
        self.name = name  # No support for phone numbers
        self.wc = 0  # Word count
        self.words = {}
        self.hours = [0 for _ in range(24)]  # Number of messages written in that hour

    def add_message(self, message, hour):
        """Adds message data to the user object"""
        self.hours[hour] += 1
        # Clear words of dots, quotation marks etc.
        for word in message.split():
            word = word.lower()
            while len(word) > 1 and word[-1] in '-,."\'!?:)—':
                word = word[:-1]
            while len(word) > 1 and word[0] in ',."\'*(-—~':
                word = word[1:]
            self.words.setdefault(word, 0)
            self.words[word] += 1
            self.wc += 1


def date_diff(msg1, msg2):
    """Calculates number of days that lie between the two given messages"""
    dt1 = date(2000+int(msg1[6:8]), int(msg1[3:5]), int(msg1[:2]))
    dt2 = date(2000+int(msg2[6:8]), int(msg2[3:5]), int(msg2[:2]))
    return (dt2 - dt1).days


def process(chat):
    """Reads chat file and prepares data for plotting"""
    data = open(chat, 'r')
    chat = data.readlines()
    data.close()
    # Initialize vars
    members = []
    first = chat[0]
    period = date_diff(first, chat[-1])
    days = [0 for _ in range(period+1)]
    # Process messages
    for line in chat:
        try:
            date = line[:8]
            hour = int(line[10:12])
            line = line[20:]
            name = line[:line.index(':')].decode('utf-8')
            line = line[line.index(': ') + 2:]
        except ValueError:
            pass  # Ignore corrupted messages
        else:
            days[date_diff(first, date)] += 1
            if all(member.name != name for member in members):
                members.append(Member(name))
            for member in members:
                if member.name == name:
                    member.add_message(line, hour)
    return members, days, period


def plot_general(members, days, period):
    """Visualizes data concerning all users"""

    # Set window title
    plt.figure().canvas.set_window_title('Whatsapp Analyzer')

    # Plot sum of messages for every day
    plt.subplot(211)
    plt.plot(days)
    plt.grid()
    plt.title('Messages Sent during %d Days' % period)
    plt.xlabel('Day since First Message')
    plt.ylabel('#Messages')

    # Plot overall message count average per hour of the day
    x = np.linspace(0, 23, num=128, endpoint=True)
    y = [e / float(period) for e in [sum([m.hours[i] for m in members]) for i in range(24)]]
    f = interp1d([i for i in range(24)], y, kind='cubic')
    plt.subplot(212)
    plt.plot(x, f(x), 'red')
    plt.xticks(np.arange(min(x), max(x)+1, 2.0))
    plt.grid()
    plt.title('Average Messages per Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('#Messages')

    # Plot highest user specific message count average per hour of the day
    y, lbs = [], []
    for i in range(24):
        mx_l, mx_m = '', -1
        for m in members:
            if m.hours[i] > mx_m:
                mx_m = m.hours[i]
                mx_l = m.name.split()[0][:1] + '.' + m.name.split()[1][:1] + '.'
        y.append(mx_m / float(period))
        lbs.append(mx_l)
    plt.scatter(range(24), y, color='blue')
    for i in range(24):
        plt.annotate(lbs[i], (i, y[i]), xytext=(5, 5), textcoords='offset points') 
    plt.legend(['All Users Together', 'Most Active User in that Hour'], loc=2)

    # Show plots
    plt.show()


def plot_user_spec(members, period):
    """Visualizes data concerning specific users"""

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
    plt.xlim([0, max(msgs)*1.15])
    plt.yticks(range(len(members)), [m.name for m in members], size='small')
    # Annotate bars
    for i in range(len(members)):
        plt.text(msgs[i]+max(msgs)*0.02, i, str(msgs[i]), ha='left', va='center')
    # Label plot and axes
    plt.title('Total Messages Sent during %d Days' % period)

    # Total message count for each member as pie chart
    m_pie = plt.subplot(222)
    # Explode max
    explode = tuple([0.1 if sum(m.hours)==max(msgs) else 0 for m in members])
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
    wc_total = [m.wc for m in members]
    explode = tuple([0.1 if m.wc==max(wc_total) else 0 for m in members])
    plt.pie(wc_total, explode=explode, labels=[' ' for m in members], colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Total Words Written as Share')
    # Configure legend
    handles, labels = w_pie.get_legend_handles_labels()
    plt.legend(handles[::-1], [m.name for m in members][::-1], loc='center', bbox_to_anchor=(0.95, 0.5))

    # Show plots
    plt.show()


def main():
    """Main function"""
    members, days, period = process(FILE)
    plot_general(members, days, period)
    #plot_user_spec(members, period)


main()

