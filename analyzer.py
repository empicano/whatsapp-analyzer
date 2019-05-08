import sys
import os
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

from matplotlib.lines import Line2D
from math import log
from random import sample
from numpy import mean


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
    # - durchschnittliche reponse zeit in sonem wellen plot (wie der wie gut ist good, well, ...)
    # - alles mit ner test-chat datei durchtesten (DONE: network)

    hours = [[0]*24 for _ in range(7)]  # messages at weekday in hour
    period = 0  # time frame of chat in days
    first = None  # date of first message
    days = []  # messages mapped on days (all users)

    def __init__(self, name):
        """Initialize member object"""
        self.name = name
        self.words = 0
        self.days = [0]*Member.period  # messages mapped on days (one user)
        self.media = 0  # number of media files sent
        self.answers = {}

    def add_message(self, message, date, predec):
        """Add data from one message to the user object"""
        Member.hours[date.weekday()][date.hour] += 1
        index = (date - Member.first).days
        Member.days[index] += 1
        self.days[index] += 1
        self.answers.setdefault(predec, 0)
        self.answers[predec] += 1
        for word in message.split():
            if word not in EXCLUDED:
                self.words += 1
            elif word == 'omitted>':
                self.media += 1


class Text:
    """Contain methods for working on the chat file"""

    @staticmethod
    def extract(line, members, predec):
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
            pass  # ignore status messages
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
            for m in members:
                if m.name == name:
                    m.add_message(line, date, predec)
                    return m.name
        return predec

    @staticmethod
    def process(path):
        """Extract and order data out of given chat file"""
        members = []
        prev = None  # previous line in chat file
        predec = None  # author of previous message (predecessor)
        with open(path) as chat:
            for line in chat:
                n = None
                if (
                    len(line) > 20
                    and line[2] == line[5] == '.'
                    and line[8:10] == ', '
                    and line[12] == line[15] == line[18] == ':'
                ):
                    if prev: predec = Text.extract(prev, members, predec)
                    prev = line
                else:
                    if prev: prev = prev[:-1] + ' ' + line
            Text.extract(prev, members, predec)
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
    months = [mean(Member.days[indexes[i]:indexes[i+1]]) for i in range(len(indexes)-1)]

    # plot total messages per day
    plt.figure()
    dates = [Member.first.date() + dt.timedelta(days=i) for i in range(Member.period)]
    s = plt.stem(dates, Member.days, markerfmt=' ', basefmt=' ', label='Total Messages per Day')[1]
    plt.setp(s, linewidth=0.5, color=COLORS[2])
    # plot overall mean of messages per day
    mn = mean(Member.days)
    plt.axhline(mn, color=COLORS[1], label='Overall Mean of Messages per Day')
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
        '{0:.{digits}f}'.format(mn, digits=2),
        xy=(Member.first.date() + dt.timedelta(days=Member.period), mn),
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
        [mean(members[i].days[k:k+7]) for k in range(index, Member.period-6, 7)]
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
        [m.words for m in members],
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
            # annotate segments with total value
            if p.get_height() > 0.03:
                ax.text(0.6, p.get_y() + shares[j] / 2, c[j], ha='center', va='center')
            # annotate segments with user names
            if i == 0:
                ax.text(-0.3, p.get_y() + shares[j] / 2, member.name, ha='right', va='center')

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
        [m.words / sum(m.days) for m in members],
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


def network(members):
    """Visualize reponse network structures.

    Display how often users answer to each other user in an alluvial
    diagram.
    """
    class LineDataUnits(Line2D):
        """Line2D taking lw argument in y axis units instead of points"""
        def __init__(self, *args, **kwargs):
            _lw_data = kwargs.pop('lw', 1)
            super().__init__(*args, **kwargs)
            self._lw_data = _lw_data

        def _get_lw(self):
            if self.axes is not None:
                ppd = 72./self.axes.figure.dpi
                trans = self.axes.transData.transform
                return ((trans((1, self._lw_data)) - trans((0, 0))) * ppd)[1]
            else:
                return 1

        def _set_lw(self, lw):
            self._lw_data = lw

        _linewidth = property(_get_lw, _set_lw)

    def ease(y0, y1):
        """Return ease in out function from point (0, y0) to (1, y1)"""
        return y0 + (y1-y0) * x**2 / (x**2 + (1-x)**2)

    fig, ax = plt.subplots()
    x = np.linspace(0.01, 0.99)
    s = sum(Member.days) - 1
    net = [[m.answers[c.name]/s if c.name in m.answers else 0 for c in members] for m in members]
    spc = 0.05  # spacing between groups
    posr = 1 + len(members)*spc
    for i in range(len(members)):
        for j, m in enumerate(members):
            posl = 1 + (len(members)-1-j)*spc - sum([sum(net[k]) for k in range(j)])
            posl -= sum(net[j][:i]) + net[j][i]
            posr -= net[j][i] + (spc if j == 0 else 0)
            # draw limitations
            p = plt.bar(0, net[j][i], 0.01, posl, color='black', align='edge').patches[0]
            plt.bar(1, net[j][i], -0.01, posr, color='black', align='edge')
            # annotate segments with user names
            if i == 0:
                tpos = 1 + len(members)*spc - spc
                tpos -= sum([sum(net[k]) for k in range(j)]) + sum(net[j])/2 + j*spc
                ax.text(-0.043, tpos, m.name, ha='right', va='center')
            # draw alluvial lines
            ax.add_line(LineDataUnits(
                x,
                ease(posl+net[j][i]/2, posr+net[j][i]/2),
                lw=net[j][i],
                alpha=0.5,
                color=COLORS[j+6]
            ))

    # set style attributes
    plt.ylim(0, 1 + len(members)*spc - spc)
    plt.title('Reponse Network')
    ax.set_axis_off()


if __name__ == '__main__':
    members = Text.process(PATH)
    # set custom plot style
    plt.style.use(os.path.join(sys.path[0], 'style.mplstyle'))
    # show plots
    for plot in [network]: #, trend, activity, shares, times, network]:
        plot(members)
        plt.gcf().canvas.set_window_title('Whatsapp Analyzer')
        plt.show(block=False)
    plt.show()

