import sys
import os
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.colors as clrs

from math import log
from random import sample
from scipy.interpolate import interp1d


# chat file
FILE = sys.argv[1] if len(sys.argv) > 1 else 'chat.txt'
# colors used when plotting message trend
TRND_THEME = ['#b0b0b0', '#07a0c3', '#e3170a']
# colors used when plotting user specific information
BACK = '#dddddd'
SPEC_THEME = ['#d3d3d3', '#a9a9a9', '#588c7e', '#f2e394', '#f2ae72', '#d96459', '#8c4646']
# colors used when plotting general user information
GNRL_THEME = ['#c7c6c1', '#5398d9', '#ff0000', '#00ff00']
# define the hour in which a day starts and ends, set 0 for start at 00:00 and end at 23:59
DAYSTART = 4

class Member:
    """Represent a chat participator"""

    # TODO
    # - make smallest first value static
    # - save len(days) as static value
    # - alle themes zu einem zusammenfassen (name: THEME)
    # - variablen in dieser Klasse umbenennen für mehr klarheit
    # - Farbe der User in den Usern speichern?
    # - member list mit dem aktivsten chatteilnehmer als erstes ordnen
    # - werden sticker erkannt?
    # - len(members) static abspeichern
    # - auf 2-8 user begrenzen, colors hardcoden, dann auch clrs import löschen
    # - den fakt, dass ein Tag von 4 bis 4 gehen soll erweitern auch auf die days list etc.

    hours = [[0 for _ in range(24)] for _ in range(7)]  # messages in hour at weekday

    def __init__(self, name, first, period):
        """Initializes object and sets variables to default values"""
        self.name = name
        self.words = {}
        self.days = [0 for _ in range(period)]  # messages mapped on days
        self.first = first  # date of first message
        self.media = 0  # number of media files sent

    def add_message(self, message, day, weekday, hour):
        """Adds message data to the user object"""
        Member.hours[weekday][hour] += 1
        self.days[day] += 1
        # excluded words
        excl = ['<image', '<video', '<‎GIF', 'omitted>']
        # strip words of dots, quotation marks etc.
        for word in message.split():
            word = word.lower()
            while len(word) > 1 and word[-1] in '*-,."\'!?:—_':
                word = word[:-1]
            while len(word) > 1 and word[0] in ',."\'*(-—~#/_':
                word = word[1:]
            if word not in excl:
                self.words.setdefault(word, 0)
                self.words[word] += 1
            elif word == 'omitted>':
                self.media += 1


class Chat:
    """Represent the chat data"""

    # TODO nur in staticmethods arbeiten und data linie für linie bearbeiten

    def __init__(self, path):
        """Initializes object and reads the chat file"""
        chfile = open(path, 'r')
        self.chat = chfile.readlines()
        chfile.close()

    @staticmethod
    def strdate(s):
        """Extract date and hour out of given string"""
        return dt.date(2000+int(s[6:8]), int(s[3:5]), int(s[:2])), int(s[10:12])

    @staticmethod
    def shftfive(date, hour):
        """Shift date so that one day starts and ends at 4 in the morning"""
        return date - dt.timedelta(days=1) if hour < 4 else date

    @staticmethod
    def idf(word, members):
        """Calculate idf value for word in members"""
        return log(len(members) / len([m for m in members if word in m.words]))

    def rnl(self):
        """Replace newline chars in messages with spaces."""
        res = []
        prev = None
        for msg in self.chat:
            # check for correct date format
            if len(msg) > 20 and msg[2] == msg[5] == '.' and msg[8:10] == ', ' and msg[12] == msg[15] == msg[18] == ':':
                if prev: res.append(prev)
                prev = msg
            else:
                # if first line is corrupted, ignore
                if prev: prev = prev[:-1] + ' ' + msg
        res.append(prev)
        self.chat = res

    def process(self):
        """Order and prepare data for plotting"""
        self.rnl()
        members = []
        first = Chat.shftfive(*Chat.strdate(self.chat[0]))
        period = (Chat.shftfive(*Chat.strdate(self.chat[-1])) - first).days + 1
        # process messages
        for line in self.chat:
            try:
                tmp = line
                hour = int(line[10:12])
                date = Chat.shftfive(*Chat.strdate(line))
                line = line[20:]
                name = line[:line.index(':')]
                line = line[line.index(': ') + 2:]
            except ValueError:
                pass  # ignore corrupted messages
            else:
                if all(member.name != name for member in members):
                    members.append(Member(name, date, period))
                for member in members:
                    if member.name == name:
                        member.add_message(line, (date-first).days, date.weekday(), hour)
        return members


def trend(members):
    """Visualize overall message count trend.

    This includes raw message count/day, mean count/day for every
    month and overall mean count/day.
    """
    period = len(members[0].days)
    days = [sum(m.days[i] for m in members) for i in range(period)]
    first = min(m.first for m in members)
    dates = [first + dt.timedelta(days=i) for i in range(period)]

    # convert from daily message count to monthly average
    start = first if first.day==1 else first.replace(day=1, month=first.month+1)
    delta_months = (
        ((first + dt.timedelta(days=period)).year - start.year) * 12
        + (first + dt.timedelta(days=period)).month
        - start.month
    )
    # get indexes of first day of every month in days list
    indexes = [(start-first).days] + [(start.replace(
        month=(start.month+i) % 12 + 1,
        year=start.year + (start.month+i) // 12
    ) - first).days for i in range(0, delta_months)]
    # get monthly messages/day mean
    months = [np.mean(days[indexes[i]:indexes[i+1]]) for i in range(len(indexes)-1)]

    # plot total messages per day
    plt.figure()
    stemline = plt.stem(dates, days, markerfmt=' ', basefmt=' ', label='Total Messages per Day')[1]
    plt.setp(stemline, linewidth=0.5, color=TRND_THEME[0])
    # plot overall mean of messages day
    mean = np.mean(days)
    plt.axhline(mean, color=TRND_THEME[2], label='Overall Mean of Messages per Day')
    # plot monthly mean of messages per day
    x = [dates[i] for i in indexes[:-1]]
    plt.plot(x, months, color=TRND_THEME[1], label='Monthly Mean of Messages per Day')

    # set style attributes
    plt.ylim(0, top=1.05*max(days))
    plt.gca().yaxis.grid(True)
    plt.legend()
    plt.title('Messages per Day')

    # annotate mean line
    plt.annotate(
        '{0:.{digits}f}'.format(mean, digits=2),
        xy=(min(m.first for m in members) + dt.timedelta(days=period-1), mean),
        xytext=(8, -3),
        textcoords='offset points',
    )
    # annotate maxima
    maxima = sorted(days, reverse=True)[:3]
    annotations = [(dates[days.index(m)], m) for m in maxima]
    for a in annotations:
        plt.annotate(
            a[0].strftime('%d.%m.%Y'),
            xy=a,
            xytext=(-10, -6),
            rotation=90,
            textcoords='offset points',
        )


def activity(members):
    """Visualize member activity over whole chat period.

    Display weekly means for every user multiple times in line charts
    emphasizing one user at a time.
    """
    period = len(members[0].days)
    days = [sum(m.days[i] for m in members) for i in range(period)]
    first = min(m.first for m in members)

    # define subplots
    fig, axarr = plt.subplots(len(members), sharex=True, sharey=True, )
    axarr = axarr[::-1]
    # compute weekly means
    index = (7 - first.weekday()) % 7
    weeks = [
        [np.mean(members[i].days[k:k+7]) for k in range(index, period, 7)]
        for i in range(len(members))
    ]
    dates = [first + dt.timedelta(days=i) for i in range(index, period, 7)]

    # plot multiple times with different emphasis
    for i in range(len(members)):
        for j in range(len(members)):
            axarr[i].plot(dates, weeks[j], color=BACK, linewidth=0.5)
        axarr[i].plot(dates, weeks[i], color=SPEC_THEME[i])
        # set style attributes
        axarr[i].yaxis.grid(True)
        axarr[i].set_ylim(0, 1.1*max([max(l) for l in weeks]))
        axarr[i].set_ylabel(members[i].name, labelpad=20, rotation=0, ha='right')

    # set title
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', left=False, bottom=False)
    plt.title('User Activity (Weekly Messages / Day Means)')


def shares(members):
    """Visualize conversation shares.

    This includes number of messages as share, number of words as share
    and average words per message.
    """
    # plot stacked bar plots visualizing shares of messages, text and media files
    fig = plt.figure()
    count = [
        [sum(m.days) for m in members],
        [sum(m.words.values()) for m in members],
        [m.media for m in members]
    ]
    for i in range(3):
        ax = fig.add_subplot(161 + i, xlim=[0, 1])
        c = count[i]
        total = sum(c)
        shares = [c / total if total else 1 / len(members) for c in c]
        for j, member in enumerate(members):
            x = plt.bar(0.6, shares[j], 0.6, bottom=sum(shares[:j]), color=SPEC_THEME[j])
            p = x.patches[0]
            # annotate segment with total value
            if p.get_height() > 0.03:
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    p.get_y() + p.get_height() / 2,
                    c[j],
                    ha='center',
                    va='center',
                )
            # annotate segments with user names
            if i == 0:
                ax.text(
                    -0.3,
                    p.get_y() + p.get_height() / 2,
                    member.name,
                    ha='right',
                    va='center'
                )

        # set style attributes
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(direction='inout', length=10)
        ax.xaxis.set_visible(False)
        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        if i: ax.set_yticklabels([])
        else: ax.set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
        ax.text(
            p.get_x() + p.get_width() / 2,
            -0.04,
            ('Messages Sent', 'Words Written', 'Media Files Sent')[i],
            ha='center'
        )

    # set title
    fig.add_subplot(121, frameon=False)
    plt.tick_params(labelcolor='none', left=False, bottom=False)
    plt.title('Shares of Messages, Words and Media Files')

    # plot average number of words and media files per message
    averages = [
        [sum(m.words.values()) / sum(m.days) for m in members],
        [m.media / sum(m.days) for m in members]
    ]
    titles = [
        'Average Words per Message',
        'Percentage of Media Files in Messages'
    ]
    for i in range(2):
        # plot overall mean
        ax = fig.add_subplot(220 + (i+1)*2, xmargin=0.05, ymargin=0.15)
        mean = sum(count[i+1]) / sum([sum(m.days) for m in members])
        plt.axvline(mean, color=TRND_THEME[2], label='Overall Mean', zorder=0)
        plt.legend()

        # plot bar chart
        plt.barh(range(len(members)), averages[i], 0.5, color=SPEC_THEME)
        plt.title(titles[i])

        # set style attributes
        ax.xaxis.grid(True)
        ax.yaxis.set_visible(False)
        if i:
            start, end = ax.get_xlim()
            ax.set_xticks([x / 100 for x in range(0, int(end*100)+1)])
            ax.set_xticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_xticks()])


def times(members):
    """Visualize message count averages in different time frames.

    This includes message count mean per hour of the day and message
    count mean per day of the week.
    """
    period = len(members[0].days)
    first = min([m.first for m in members])

    fwd = first.weekday()  # weekday of first message
    lwd = (first + dt.timedelta(days=period-1)).weekday()  # weekday of last message
    week_count = (period - (7-fwd) - (lwd+1)) / 7
    weekday_counts = [(week_count + (i >= fwd) + (i <= lwd)) for i in range(7)]

    # plot message count mean per hour of the day (whole week)
    fig = plt.figure()
    ax = fig.add_subplot(211, xmargin=0.1, ymargin=0.1)
    weekdays = [sum(Member.hours[i]) for i in range(7)]
    means = list(map(lambda w, c: w / c, weekdays, weekday_counts))
    for i in range(7):
        plt.plot((i*24, (i+1)*24), (means[i]/24,)*2, color=TRND_THEME[2])
    raw = [e / weekday_counts[i] for i, h in enumerate(Member.hours) for e in h]
    plt.plot(range(24*7+1), raw[DAYSTART:] + raw[:DAYSTART+1])

    # set style attributes
    ax.grid(True)
    ticks = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    plt.xticks(range(0, 24*8, 24), ticks)
    plt.title('Average Message Count per Hour of the Day (Whole Week)')
    patch, = plt.plot([], color=TRND_THEME[2], label='Daily Mean')
    plt.legend(handles=[patch])

    # plot message count mean per hour of the day (one day)
    ax = fig.add_subplot(212)
    x = np.linspace(DAYSTART, DAYSTART+24, num=1000)
    labels = ['Overall', 'Midweek (MTWT)', 'Weekend (FSS)']
    ranges = [range(7), range(4), range(4, 7)]
    for i in range(3):
        totals = [sum([Member.hours[day][hour] for day in ranges[i]]) for hour in range(24)]
        div = sum([weekday_counts[x] for x in ranges[i]])
        means = [x / div for x in totals]
        # interpolate over longer interval to ensure that end points have same slopes
        means = means * 4
        # cubic interpolate
        f = interp1d(range(-24, 72), means, kind='cubic')
        plt.plot(x, f(x), GNRL_THEME[i+1], lw=(1 if i else 3), ls=('--' if i else '-'), label=labels[i])

    # set style attributes
    ax.grid(True)
    scope = max(f(x)) - min(f(x))
    plt.ylim(-0.1*scope, max(f(x))+0.1*scope)
    plt.xticks(range(DAYSTART, DAYSTART+25), list(range(DAYSTART, 24)) + list(range(DAYSTART+1)))
    plt.title('Average Message Count per Hour of the Day (One Day)')
    plt.legend()


def worduse_md(members, path='worduse.md'):
    """Generates markdown document with most used and most important (tf-idf) words for each user"""
    with open(path, 'w') as mdfile:
        for m in sorted(members, key=lambda e: len(e.words), reverse=True):
            mdfile.write('# ' + m.name + '\nMost used words | Frequency in messages | Most important (tf-idf) words | Frequency in messages\n-|-|-|-\n')
            most_used = sorted(m.words.items(), key=lambda e: e[1], reverse=True)[:15]
            avg_per_msg_used = [wd[1] / sum(m.days) for wd in most_used]
            most_impt = sorted(m.words.items(), key=lambda x: x[1] * Chat.idf(x[0], members), reverse=True)[:15]
            avg_per_msg_impt = [wd[1] / sum(m.days) for wd in most_impt]
            for wd_used, avg_used, wd_impt, avg_impt in zip(most_used, avg_per_msg_used, most_impt, avg_per_msg_impt):
                mdfile.write(wd_used[0] + ' | ' + '{0:.3f}'.format(avg_used) + ' | ' + wd_impt[0] + '| {0:.3f}'.format(avg_impt) + '\n')


if __name__ == '__main__':
    chat = Chat(FILE)
    members = sorted(chat.process(), key=lambda m: sum(m.days))
    SPEC_THEME = SPEC_THEME[len(SPEC_THEME)-len(members):] if len(members) <= len(SPEC_THEME) else sample(list(clrs.cnames), len(members))

    # set custom plot style
    plt.style.use(os.path.join(sys.path[0], 'style.mplstyle'))

    # show plots
    plots = [times]  # [trend, activity, shares, times]
    for plot in plots:
        plot(members)
        plt.gcf().canvas.set_window_title('Whatsapp Analyzer')
        plt.show(block=False)
    plt.show()

    # Uncomment for generating a markdown file containing worduse statistics
    # worduse_md(members)

