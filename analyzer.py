#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
import matplotlib.pyplot as plt, matplotlib.colors as clrs, matplotlib.dates as mdts
import datetime as dt

from scipy.interpolate import interp1d

# exported chat file
FILE = 'chat.txt'
# colors used when plotting user specific information
SPEC_THEME = ['#d3d3d3', '#a9a9a9', '#588c7e', '#f2e394', '#f2ae72', '#d96459', '#8c4646']
# colors used when plotting general information
GNRL_THEME = ['#14325c', '#c9c9c9', '#5398d9', '#ff0000']


class Member:
    """Represents one chat participator"""

    def __init__(self, name, first):
        """Initializes object and sets variables to default values"""
        self.name = name
        self.wc = 0  # word count
        self.words = {}
        self.hours = [0 for _ in range(24)]  # number of messages written in that hour
        self.first = first  # date of first message

    def add_message(self, message, hour):
        """Adds message data to the user object"""
        self.hours[hour] += 1
        # clear words of dots, quotation marks etc.
        for word in message.split():
            word = word.lower()
            while len(word) > 1 and word[-1] in '-,."\'!?:)—':
                word = word[:-1]
            while len(word) > 1 and word[0] in ',."\'*(-—~':
                word = word[1:]
            self.words.setdefault(word, 0)
            self.words[word] += 1
            self.wc += 1


def string_to_date(s):
    """Converts string of the format dd.mm.yy to a datetime object"""
    return dt.date(2000+int(s[6:8]), int(s[3:5]), int(s[:2]))


def date_diff(msg1, msg2):
    """Calculates number of days that lie between two given messages"""
    return (string_to_date(msg2) - string_to_date(msg1)).days


def rm_newline_chars(chat):
    """Removes newline chars from messages"""
    res = []
    prev = None
    for line in chat:
        try:
            int(line[:2])
            int(line[3:5])
            int(line[6:8])
            if not (line[2] == line[5] == '.') or line[8] != ',':
                raise ValueError
            if prev: res.append(prev)
            prev = line
        except ValueError:
            prev = prev[:-1] + ' ' + line
    res.append(prev)
    return res


def process(chat):
    """Reads chat file and prepares data for plotting"""
    data = open(chat, 'r')
    chat = data.readlines()
    data.close()
    # remove newline chars from messages
    chat = rm_newline_chars(chat)
    # initialize vars
    members = []
    first = chat[0]
    period = date_diff(first, chat[-1]) + 1
    days = [0 for _ in range(period)]
    # process messages
    for line in chat:
        try:
            date = line[:8]
            hour = int(line[10:12])
            line = line[20:]
            name = line[:line.index(':')]
            line = line[line.index(': ') + 2:]
        except ValueError:
            pass  # ignore corrupted messages
        else:
            days[date_diff(first, date)] += 1
            if all(member.name != name for member in members):
                members.append(Member(name, string_to_date(date)))
            for member in members:
                if member.name == name:
                    member.add_message(line, hour)
    return members, days


def plot_general(members, days, period):
    """Visualizes data concerning all users"""

    # set window title
    plt.figure().canvas.set_window_title('Whatsapp Analyzer')

    # plot monthly average of messages per day
    plt.subplot(211)
    # set up date xlables
    dates = [min(m.first for m in members) + dt.timedelta(days=i) for i in range(period)]
    plt.gca().xaxis.set_major_formatter(mdts.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdts.MonthLocator(interval=(period // 300) or 1))
    # convert from daily message count to monthly average
    first = min(m.first for m in members)
    start = first if first.day==1 else first.replace(day=1, month=first.month+1)
    m_diff = ((first + dt.timedelta(days=period)).year - start.year) * 12 + (first + dt.timedelta(days=period)).month - start.month
    idxs = [(start.replace(month=(start.month+i) % 12 + 1, year=start.year + ((start.month+i) // 12)) - first).days for i in range(0, m_diff)]
    idxs.insert(0, (start-first).days)
    x = [dates[i] for i in idxs[:-1]]
    months = [np.mean(days[idxs[i]:idxs[i+1]]) for i in range(len(idxs)-1)]
    # plot bars
    plt.bar(x, months, [idxs[i]-idxs[i-1] for i in range(1, len(idxs))], color=GNRL_THEME[1], align='edge')

    # plot message count on all days
    plt.plot(dates, days, GNRL_THEME[0])
    plt.ylim([0, max(days)*1.1])
    plt.gca().yaxis.grid(True)
    plt.legend(['Total Number on specific Day', 'Average in that Month'], loc=2)
    plt.title('Messages per Day')
    plt.ylabel('#Messages')
    # annotate maxima
    mxma = []
    cp = days[:]
    for i in range(3):
        mxma.append((dates[days.index(max(cp))], max(cp)))
        cp.remove(max(cp))
    for mxm in mxma:
        plt.annotate(mxm[0].strftime('%a, %d.%m.%Y'), xy=mxm,
                     xytext=(30, 0), textcoords='offset points', verticalalignment='center', arrowprops=dict(arrowstyle='->'))

    # plot overall message count average per hour of the day
    x = np.linspace(0, 23, num=128, endpoint=True)
    y = [e / period for e in [sum([m.hours[i] for m in members]) for i in range(24)]]
    # cubic interpolate
    f = interp1d([i for i in range(24)], y, kind='cubic')
    plt.subplot(212)
    plt.plot(x, f(x), GNRL_THEME[2])
    plt.xticks(np.arange(min(x), max(x)+1, 2.0))
    plt.grid()
    plt.title('Average Messages per Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('#Messages')

    # plot average number of messages of most active user in that hour
    y, lbs = [], []
    for i in range(24):
        mx_l, mx_m = '', -1
        for m in members:
            if m.hours[i] > mx_m:
                mx_m = m.hours[i]
                mx_l = m.name.split()[0][:1] + '.' + m.name.split()[1][:1] + '.'
        y.append(mx_m / period)
        lbs.append(mx_l)
    plt.scatter(range(24), y, color=GNRL_THEME[3])
    # annotate points with initials
    for i in range(24):
        plt.annotate(lbs[i], xy=(i, y[i]), xytext=(0, 10), textcoords='offset points')
    plt.legend(['All Users Together', 'Most Active User in that Hour'], loc=2)

    # show plots
    plt.show()


def plot_users(members, period):
    """Visualizes data concerning specific users"""

    # set window title
    plt.figure().canvas.set_window_title('Whatsapp Analyzer')
    # set colors
    colors = SPEC_THEME[:len(members)] if len(members) <= len(SPEC_THEME) else random.sample(clrs.cnames, len(members))

    # total message count for each member as bar graph
    members = sorted(members, key=lambda m: sum(m.hours))
    plt.subplot(221)
    msgs = [sum(m.hours) for m in members]
    barlst = plt.barh(range(len(members)), msgs, align='center', height=0.4)
    # set bar colors
    for i in range(len(barlst)):
        barlst[i].set_color(colors[i])
    plt.xlim([0, max(msgs)*1.15])
    plt.yticks(range(len(members)), [m.name for m in members], size='small')
    # annotate bars with exakt value
    for i in range(len(members)):
        plt.text(msgs[i]+max(msgs)*0.02, i, str(msgs[i]), ha='left', va='center')
    plt.title('Total Messages Sent during %d Days' % period)

    # total message count for each member as pie chart
    m_pie = plt.subplot(222)
    # explode max
    explode = tuple([0.1 if sum(m.hours)==max(msgs) else 0 for m in members])
    plt.pie(msgs, explode=explode, labels=[' ' for m in members], colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Total Messages Sent as Share')
    # configure legend
    handles, labels = m_pie.get_legend_handles_labels()
    plt.legend(handles[::-1], [m.name for m in members][::-1], loc='center', bbox_to_anchor=(0.95, 0.5))

    # words per message for each member as bar graph
    plt.subplot(223)
    wc_avg = [m.wc / sum(m.hours) for m in members]
    barlst = plt.barh(range(len(members)), wc_avg, align='center', height=0.4)
    # set bar colors
    for i in range(len(barlst)):
        barlst[i].set_color(colors[i])
    plt.xlim([0, max(wc_avg)*1.15])
    plt.yticks(range(len(members)), [m.name for m in members], size='small')
    # annotate bars exact value
    for i in range(len(members)):
        plt.text(wc_avg[i]+max(wc_avg)*0.02, i, str(round(wc_avg[i], 3)), ha='left', va='center')
    plt.title('Average Words per Message')

    # total word count for each member as pie chart
    w_pie = plt.subplot(224)
    # explode max
    wc_total = [m.wc for m in members]
    explode = tuple([0.1 if m.wc==max(wc_total) else 0 for m in members])
    plt.pie(wc_total, explode=explode, labels=[' ' for m in members], colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Total Words Written as Share')
    # configure legend
    handles, labels = w_pie.get_legend_handles_labels()
    plt.legend(handles[::-1], [m.name for m in members][::-1], loc='center', bbox_to_anchor=(0.95, 0.5))

    # show plots
    plt.show()


def main():
    """Main function"""
    members, days = process(FILE)
    period = len(days)
    plot_general(members, days, period)
    plot_users(members, period)


main()

