import sys
import os
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

from math import log
from random import sample
from scipy.interpolate import interp1d


# path to chat file
PATH = sys.argv[1]
# define the hour in which a day starts and ends, set 0 for start at 00:00 and end at 23:59
DAYSTART = 4
# interval to consider
BOUND = [None, None]
if len(sys.argv) > 3:
    for i in range(len(BOUND)):
        BOUND[i] = dt.datetime.strptime(sys.argv[i+2], '%d.%m.%Y')
        BOUND[i] = BOUND[i] - dt.timedelta(days=1) if BOUND[i].hour < DAYSTART else BOUND[i]
# words excluded in all statistics
EXCLUDED = ['<image', '<video', '<‎GIF', 'omitted>']
# color scheme
COLORS = [
    '#235789',  # line plots
    '#e3170a',  # mean lines
    '#bbbbbb',  # stems in trend plot
    '#dddddd',  # background lines in activity plot
    '#ffa630',  # midweek line in times plot
    '#87bba2',  # weekend line in times plot
    '#1b9e77',  # user 1
    '#d95f02',  # user 2
    '#7570b3',  # user 3
    '#e7298a',  # user 4
    '#66a61e',  # user 5
    '#e6ab01',  # user 6
    '#a6761d',  # user 7
    '#666666',  # user 8
]
# display up to this number of users, if greater, add up the rest and display as one
MEMBERMAX = 8


class Member:
    """Represent a chat participator"""

    # TODO
    # - werden sticker erkannt?
    # - times mal ohne interpolate ausprobieren

    hours = [[0]*24 for _ in range(7)]  # messages at weekday in hour
    period = 0  # time frame of chat in days
    first = None  # date of first message
    days = []  # messages mapped on days (all users)

    def __init__(self, name):
        """Initialize member object"""
        self.name = name
        self.words = {}
        self.days = [0]*Member.period  # messages mapped on days (one user)
        self.media = 0  # number of media files sent

    def add_message(self, message, date):
        """Add data from one message to the user object"""
        Member.hours[date.weekday()][date.hour] += 1
        index = (date - Member.first).days
        Member.days[index] += 1
        self.days[index] += 1
        for word in message.split():
            word = Text.strip(word)
            if word and word not in EXCLUDED:
                self.words.setdefault(word, 0)
                self.words[word] += 1
            elif word == 'omitted>':
                self.media += 1


class Text:
    """Contain methods for working on the chat file"""

    @staticmethod
    def strip(word):
        """Strip word of dots, quotation marks etc."""
        word = word.lower()
        chars = ',."\'*-~—_/#:;!?&()'
        while word and word[-1] in chars:
            word = word[:-1]
        while word and word[0] in chars:
            word = word[1:]
        return word

    @staticmethod
    def extract(line, members):
        """Extract data out of one line"""
        try:
            date = dt.datetime(
                2000+int(line[6:8]),
                int(line[3:5]),
                int(line[:2]),
                hour=int(line[10:12])
            )
            # shift date according to DAYSTART
            date = date - dt.timedelta(days=1) if date.hour < DAYSTART else date
            if BOUND[0] and not (BOUND[0] <= date < BOUND[1]): return
            if not Member.first:
                if BOUND[0]: Member.first = BOUND[0]
                else: Member.first = date.replace(hour=DAYSTART)
            line = line[20:]
            name, line = line.split(': ', 1)
        except ValueError:
            pass  # ignore status messages and corrupted messages
        else:
            # check if we have to change the index in the days list
            while (max(date, BOUND[1] if BOUND[1] else date) - Member.first).days >= Member.period:
                Member.period += 1
                Member.days.append(0)
                for member in members:
                    member.days.append(0)
            # add data to member object
            if all(member.name != name for member in members):
                members.append(Member(name))
            for member in members:
                if member.name == name: member.add_message(line, date)

    @staticmethod
    def process(path):
        """Extract and order data out of given chat file"""
        members = []
        prev = None
        with open(path) as chat:
            for line in chat:
                if (
                    len(line) > 20
                    and line[2] == line[5] == '.'
                    and line[8:10] == ', '
                    and line[12] == line[15] == line[18] == ':'
                ):
                    if prev: Text.extract(prev, members)
                    prev = line
                else:
                    if prev: prev = prev[:-1] + ' ' + line
            Text.extract(prev, members)
        members = sorted(members, key=lambda m: sum(m.days), reverse=True)
        # if number of members is greater than MEMBERMAX, add up the rest
        if len(members) > MEMBERMAX:
            others = Member('Others')
            for m in members[MEMBERMAX-1:]:
                for i, d in enumerate(m.days):
                    others.days[i] += d
                others.media += m.media
                for word, count in m.words.items():
                    others.words.setdefault(word, 0)
                    others.words[word] += count
            members = members[:MEMBERMAX-1] + [others]
        return members

    '''
    # TODO wenn wir das wegmachen auch das word dictionary und die strip function!
    @staticmethod
    def idf(word, members):
        """Calculate idf value for word in members"""
        return log(len(members) / len([m for m in members if word in m.words]))
    '''


def trend(members):
    """Visualize overall message count trend.

    This includes raw message count/day, mean count/day for every
    month and overall mean count/day.
    """
    # convert from daily message count to monthly average
    start = (
        Member.first if Member.first.day==1
        else Member.first.replace(
                day=1,
                month=Member.first.month%12+1,
                year=Member.first.year+1 if Member.first.month==12 else Member.first.year
        )
    )
    last = Member.first + dt.timedelta(days=Member.period)
    delta_months = (last.year - start.year) * 12 + last.month - start.month
    # get indexes of first day of every month in days list
    indexes = [(start-Member.first).days] + [(start.replace(
        month=(start.month+i) % 12 + 1,
        year=start.year + (start.month+i) // 12
    ) - Member.first).days for i in range(0, delta_months)]
    # get monthly messages/day mean
    months = [np.mean(Member.days[indexes[i]:indexes[i+1]]) for i in range(len(indexes)-1)]

    # plot total messages per day
    plt.figure()
    dates = [Member.first.date() + dt.timedelta(days=i) for i in range(Member.period)]
    s = plt.stem(dates, Member.days, markerfmt=' ', basefmt=' ', label='Total Messages per Day')[1]
    plt.setp(s, linewidth=0.5, color=COLORS[2])
    # plot overall mean of messages per day
    mean = np.mean(Member.days)
    plt.axhline(mean, color=COLORS[1], label='Overall Mean of Messages per Day')
    # plot monthly mean of messages per day
    x = [dates[i] for i in indexes[:-1]]
    plt.plot(x, months, color=COLORS[0], label='Monthly Mean of Messages per Day')

    # set style attributes
    plt.xlim(
        Member.first.date() - dt.timedelta(days=1),
        Member.first.date() + dt.timedelta(Member.period)
    )
    plt.ylim(0, 1.05*max(Member.days))
    plt.gca().yaxis.grid(True)
    plt.legend()
    plt.title('Messages per Day (Over a Period of ' + str(Member.period) + ' Days)')
    # set formatter and locator (autolocator has problems setting good date xticks)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=(Member.period // 250) or 1))
    plt.gca().callbacks.connect(
        'xlim_changed',
        lambda ax: ax.xaxis.set_major_locator(
            mdates.MonthLocator(interval=(int(ax.get_xlim()[1] - ax.get_xlim()[0]) // 250) or 1)
        )
    )

    # annotate mean line
    plt.annotate(
        '{0:.{digits}f}'.format(mean, digits=2),
        xy=(Member.first.date() + dt.timedelta(days=Member.period), mean),
        xytext=(8, -3),
        textcoords='offset points',
    )
    # annotate maxima
    maxima = sorted(Member.days, reverse=True)[:3]
    annotations = [(dates[Member.days.index(m)], m) for m in maxima]
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

    Display weekly means for every user in a spaghetti plot emphasizing
    one user at a time.
    """
    # compute weekly means
    fig, axarr = plt.subplots(len(members), sharex=True, sharey=True, squeeze=False)
    axarr = [ax for lt in axarr for ax in lt]
    index = (7 - Member.first.weekday()) % 7
    weeks = [
        [np.mean(members[i].days[k:k+7]) for k in range(index, Member.period-6, 7)]
        for i in range(len(members))
    ]
    dates = [Member.first.date() + dt.timedelta(days=i) for i in range(index, Member.period-6, 7)]

    # plot multiple times with different emphasis
    for i, member in enumerate(members):
        for j in range(len(members)):
            axarr[i].plot(dates, weeks[j], color=COLORS[3], linewidth=0.5)
        axarr[i].plot(dates, weeks[i], color=COLORS[i+6])
        # set style attributes
        axarr[i].yaxis.grid(True)
        if weeks[0]: axarr[i].set_ylim(0, 1.1*max([max(l) for l in weeks]))
        axarr[i].set_ylabel(member.name, labelpad=20, rotation=0, ha='right')
        plt.xlim(
            Member.first.date() - dt.timedelta(days=1),
            Member.first.date() + dt.timedelta(Member.period)
        )
        # set formatter and locator (autolocator has problems setting good date xticks)
        axarr[i].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        axarr[i].xaxis.set_major_locator(mdates.MonthLocator(interval=(Member.period // 250) or 1))
        axarr[i].callbacks.connect(
            'xlim_changed',
            lambda ax: ax.xaxis.set_major_locator(
                mdates.MonthLocator(interval=(int(ax.get_xlim()[1] - ax.get_xlim()[0]) // 250) or 1)
            )
        )

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
    members = members[::-1]
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
            x = plt.bar(0.6, shares[j], 0.6, bottom=sum(shares[:j]), color=COLORS[len(members)-j+5])
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
    plt.title('Shares of Messages, Words and Media Files per User')

    # plot average number of words and media files per message
    averages = [
        [sum(m.words.values()) / sum(m.days) for m in members],
        [m.media / sum(m.days) for m in members]
    ]
    titles = [
        'Average Words per Message',
        'Share of Media Files in Messages'
    ]
    for i in range(2):
        # plot overall mean
        ax = fig.add_subplot(220 + (i+1)*2, xmargin=0.05, ymargin=0.15)
        mean = sum(count[i+1]) / sum(Member.days)
        plt.axvline(mean, color=COLORS[1], label='Overall Mean', zorder=0)
        plt.legend()
        # plot bar chart
        plt.barh(range(len(members)), averages[i], 0.5, color=COLORS[5+len(members):5:-1])
        plt.title(titles[i])
        # set style attributes
        ax.xaxis.grid(True)
        ax.yaxis.set_visible(False)
        if i: ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: str(x*100) + '%'))


def times(members):
    """Visualize message count averages in different time frames.

    This includes message count mean per hour of the day and message
    count mean per day of the week.
    """
    weekday_counts = [0]*7
    for i in range(Member.period):
        weekday_counts[(Member.first + dt.timedelta(days=i)).weekday()] += 1

    # plot message count mean per hour of the day (whole week)
    fig = plt.figure()
    ax = fig.add_subplot(211, xmargin=0.05, ymargin=0.1)
    weekdays = [sum(Member.hours[i]) for i in range(7)]
    means = list(map(lambda w, c: w / c if c else 0, weekdays, weekday_counts))
    for i in range(7):
        plt.plot((i*24, (i+1)*24), (means[i]/24,)*2, color=COLORS[1])
    raw = [e / c if c else 0 for h, c in zip(Member.hours, weekday_counts) for e in h]
    plt.plot(range(24*7+1), raw[DAYSTART:] + raw[:DAYSTART+1], COLORS[0])

    # set style attributes
    ax.grid(True)
    ticks = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    plt.xticks(range(0, 24*8, 24), [s + ' ' + str(DAYSTART).zfill(2) + ':00' for s in ticks])
    plt.title('Average Message Count per Hour of the Day (Whole Week)')
    patch, = plt.plot([], color=COLORS[1], label='Daily Mean')
    plt.legend(handles=[patch])

    # plot message count mean per hour of the day (one day)
    ax = fig.add_subplot(212, xmargin=0.05)
    labels = ['Overall', 'Midweek (MTWT)', 'Weekend (FSS)']
    ranges = [range(7), range(4), range(4, 7)]
    maximum = 0
    for i in range(3):
        totals = [sum([Member.hours[day][hour] for day in ranges[i]]) for hour in range(24)]
        div = sum(weekday_counts[x] for x in ranges[i])
        means = [x / div if div else 0 for x in totals]
        means.append(means[0])
        if i: plt.plot(range(25), means, COLORS[i+3], lw=1.5, ls=':', label=labels[i])
        else: plt.plot(range(25), means, COLORS[0], label=labels[0], zorder=10)
        maximum = max(maximum, max(means))

    # set style attributes
    ax.grid(True)
    plt.ylim(-0.1*maximum, 1.1*maximum)
    plt.xticks(range(25), list(range(DAYSTART, 24)) + list(range(DAYSTART+1)))
    plt.title('Average Message Count per Hour of the Day (One Day)')
    plt.legend()


'''
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
'''


if __name__ == '__main__':
    members = Text.process(PATH)
    # set custom plot style
    plt.style.use(os.path.join(sys.path[0], 'style.mplstyle'))
    # show plots
    for plot in [times]:  # trend, activity, shares, times]:
        plot(members)
        plt.gcf().canvas.set_window_title('Whatsapp Analyzer')
        plt.show(block=False)
    plt.show()

