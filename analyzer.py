import sys
import numpy as np
import datetime as dt

import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.dates as mdts
import matplotlib.lines as lns

from random import sample
from scipy.interpolate import interp1d


# exported chat file
FILE = sys.argv[1] if len(sys.argv) > 1 else '_chat.txt'
# colors used when plotting message trend
TRND_THEME = ['#14325c', '#c7c6c1']
# colors used when plotting general user information
GNRL_THEME = ['#c7c6c1', '#5398d9', '#ff0000', '#00ff00']
# colors used when plotting user specific information
SPEC_THEME = ['#d3d3d3', '#a9a9a9', '#588c7e', '#f2e394', '#f2ae72', '#d96459', '#8c4646']


class Member:
    """Represents chat participator"""

    def __init__(self, name, first, period):
        """Initializes object and sets variables to default values"""
        self.name = name
        self.words = {}
        self.hours_mg = [[0 for _ in range(24)] for _ in range(7)]  # messages in hour at weekday
        self.hours_wd = [[0 for _ in range(24)] for _ in range(7)]  # words in hour at weekday
        self.days = [0 for _ in range(period)]  # messages mapped on days
        self.first = first  # date of first message

    def add_message(self, message, day, weekday, hour):
        """Adds message data to the user object"""
        self.hours_mg[weekday][hour] += 1
        self.days[day] += 1
        # strip words of dots, quotation marks etc.
        for word in message.split():
            self.hours_wd[weekday][hour] += 1
            word = word.lower()
            while len(word) > 1 and word[-1] in '-,."\'!?:)—':
                word = word[:-1]
            while len(word) > 1 and word[0] in ',."\'*(-—~#':
                word = word[1:]
            self.words.setdefault(word, 0)
            self.words[word] += 1


class Chat:
    """Represents the chat data"""

    def __init__(self, path):
        """Initializes object and reads the chat file"""
        chfile = open(path, 'r')
        self.chat = chfile.readlines()
        chfile.close()

    @staticmethod
    def strdate(s):
        """Extracts date and hour out of given string"""
        return dt.date(2000+int(s[6:8]), int(s[3:5]), int(s[:2])), int(s[10:12])

    @staticmethod
    def shftfive(date, hour):
        """Shifts date so that one day starts and ends at 4 in the morning"""
        return date - dt.timedelta(days=1) if hour < 4 else date

    def _rmnl(self):
        """Removes newline chars from messages"""
        res = []
        prev = None
        for msg in self.chat:
            # check for correct formatting
            if len(msg) > 20 and msg[2] == msg[5] == '.' and msg[8:10] == ', ' and msg[12] == msg[15] == msg[18] == ':':
                if prev: res.append(prev)
                prev = msg
            else:
                prev = prev[:-1] + ' ' + msg
        res.append(prev)
        self.chat = res

    def process(self):
        """Orders and prepares data for plotting"""
        self._rmnl()
        # initialize vars
        members = []
        first = Chat.shftfive(*Chat.strdate(self.chat[0]))
        period = (Chat.shftfive(*Chat.strdate(self.chat[-1])) - first).days + 1
        # process messages
        for line in self.chat:
            try:
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


def plot_trends(members):
    """Visualizes trends over the whole chat period"""

    def msgtrend():
        """Visualizes overall message trend"""

        # set up date xlables
        plt.gca().xaxis.set_major_formatter(mdts.DateFormatter('%b %Y'))
        plt.gca().xaxis.set_major_locator(mdts.MonthLocator(interval=(period // 300) or 1))
        # convert from daily message count to monthly average
        first = min(m.first for m in members)
        start = first if first.day==1 else first.replace(day=1, month=first.month+1)
        m_diff = ((first + dt.timedelta(days=period)).year - start.year) * 12 + (first + dt.timedelta(days=period)).month - start.month
        idxs = [(start-first).days] + [(start.replace(month=(start.month+i) % 12 + 1, year=start.year + ((start.month+i) // 12)) - first).days for i in range(0, m_diff)]
        x = [dates[i] for i in idxs[:-1]]
        months = [np.mean(days[idxs[i]:idxs[i+1]]) for i in range(len(idxs)-1)]
        # plot monthly average of messages per day
        plt.subplot(211)
        plt.bar(x, months, [idxs[i]-idxs[i-1] for i in range(1, len(idxs))], color=TRND_THEME[1], align='edge')
        # plot message count on all days
        plt.plot(dates, days, TRND_THEME[0])
        # limiters, legend, labels
        plt.xlim([dates[0]-dt.timedelta(days=period*0.03), dates[period-1]+dt.timedelta(days=period*0.03)])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().tick_params(top='off', right='off')
        plt.legend(['Total Number on Specific Day', 'On Average in that Month'], loc=2)
        plt.title('Messages per Day', y=1.03, weight='bold')
        plt.ylabel('# Messages')
        # annotate maxima
        mxma = []
        cp = days[:]
        for i in range(3):
            mxma.append((dates[days.index(max(cp))], max(cp)))
            cp.remove(max(cp))
        for mxm in mxma:
            lbl = mxm[0].strftime('%a, %d.%m.%Y')
            plt.annotate(lbl, xy=mxm, xytext=(-10, -6), rotation=90, textcoords='offset points', size='small')

    def activity():
        """Visualizes member activity over whole chat period"""

        # set up date xlables
        plt.gca().xaxis.set_major_formatter(mdts.DateFormatter('%b %Y'))
        plt.gca().xaxis.set_major_locator(mdts.MonthLocator(interval=(period // 300) or 1))
        # plot activity period
        plt.subplot(212)
        for i in range(len(members)):
            plt.scatter(dates, [i if d else None for d in members[i].days], color=SPEC_THEME[i], alpha=.4, s=tuple(map(lambda e: e/max(days)*1500, members[i].days)))
        # limiters, legend, labels
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().tick_params(top='off', right='off')
        plt.gca().yaxis.grid(False)
        plt.gca().set_yticklabels([''] + [m.name for m in members])
        plt.xlim([dates[0]-dt.timedelta(days=period*0.03), dates[period-1]+dt.timedelta(days=period*0.03)])
        plt.ylim([-1, len(members)])
        plt.title('Activity Period', y=1.03, weight='bold')

    # set up data
    period = len(members[0].days)
    days = [sum(m.days[i] for m in members) for i in range(period)]
    dates = [min(m.first for m in members) + dt.timedelta(days=i) for i in range(period)]
    # plot
    msgtrend()
    activity()
    # configure spacing
    plt.subplots_adjust(wspace=0.20, hspace=0.40)


def plot_usetime(members):
    """Visualizes data concerning all users"""

    def weekdayavg():
        """plot message count average on specific day of the week"""

        wd_sum_msgs = [sum([sum(m.hours_mg[i]) for m in members]) for i in range(7)]
        frst = min([m.first for m in members])
        frst_wd = frst.weekday()  # weekday of first message
        last_wd = (frst + dt.timedelta(days=period-1)).weekday()  # weekday of last message
        wd_count = [(period - last_wd - 1) // 7 for _ in range(7)]
        for i in range(7, (frst_wd if frst_wd else 7), -1):
            wd_count[i-1] += 1
        for i in range(last_wd+1):
            wd_count[i] += 1
        wd_avg_msgs = tuple(map(lambda e, a: e / a, wd_sum_msgs, wd_count))
        plt.subplot(221)
        plt.bar(range(7), wd_avg_msgs, align='center', color=GNRL_THEME[0])
        # limiters, xticks, labels
        wds = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        plt.xticks(range(7), wds)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().tick_params(top='off', right='off')
        plt.gca().xaxis.grid(False)
        plt.title('Average Messages per Weekday', y=1.03, weight='bold')
        plt.ylabel('# Messages')

    def houravg():
        """Visualizes message count average on specific hour of the day"""

        x = np.linspace(0, 24, num=180, endpoint=True)
        # days are defined to start at 4 in the morning
        overall = [e / period for e in (sum((sum((m.hours_mg[i][j] for i in range(7))) for m in members)) for j in range(24))]
        overall = overall[4:] + overall[:5]
        # get midweek (mon, tue, wen, thu) hours
        midweek = [e / (period*4/7) for e in (sum((sum((m.hours_mg[i][j] for i in range(4))) for m in members)) for j in range(24))]
        midweek = midweek[4:] + midweek[:5]
        # get weekend (fri, sat, sun) hours
        weekend = [e / (period*3/7) for e in (sum((sum((m.hours_mg[i][j] for i in range(4, 7))) for m in members)) for j in range(24))]
        weekend = weekend[4:] + weekend[:5]
        # cubic interpolate
        f = interp1d(range(25), overall, kind='cubic')
        g = interp1d(range(25), midweek, kind='cubic')
        h = interp1d(range(25), weekend, kind='cubic')
        # plot
        plt.subplot(212)
        plt.plot(x, f(x), GNRL_THEME[1], ls='-', lw=2)
        plt.plot(x, g(x), GNRL_THEME[2], ls='--', lw=2)
        plt.plot(x, h(x), GNRL_THEME[3], ls='--', lw=2)
        # limiters, ticks, labels, legend
        plt.xticks(range(25), [i for i in range(4, 24)] + [i for i in range(5)])
        plt.xlim([-1, 25])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().tick_params(top='off', right='off')
        plt.title('Average Messages per Hour of the Day', y=1.03, weight='bold')
        plt.ylabel('# Messages')
        plt.legend(['Overall', 'Midweek (Mo,Tu,We,Th)', 'Weekend (Fr,Sa,Su)'], loc=2)

    # set up general data
    period = len(members[0].days)
    # plot
    weekdayavg()
    houravg()
    # adjust spacing
    plt.subplots_adjust(wspace=0.20, hspace=0.40)


def plot_users(members):
    """Visualizes data concerning specific users"""

    def msgcount():
        """Total message count for each member as bar graph"""
        plt.subplot(221)
        period = len(members[0].days)
        barlst = plt.barh(range(len(members)), msgs, align='center', height=0.5)
        # set bar colors
        for i in range(len(barlst)):
            barlst[i].set_color(SPEC_THEME[i])
        plt.xlim([0, max(msgs)*1.15])
        plt.yticks(range(len(members)), (m.name for m in members))
        plt.gca().yaxis.grid(False)
        plt.gca().tick_params(top='off', right='off')
        # annotate bars with exakt value
        for i in range(len(members)):
            plt.text(msgs[i]+max(msgs)*0.02, i, str(msgs[i]), ha='left', va='center')
        plt.title('Total Messages Sent during %d Days' % period, y=1.03, weight='bold')

    def msgcountpie():
        """Total message count for each member as pie chart"""
        m_pie = plt.subplot(222)
        # explode max
        explode = tuple(0.1 if sum(m.days)==max(msgs) else 0 for m in members)
        plt.pie(msgs, explode=explode, labels=[' ' for m in members], colors=SPEC_THEME, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Total Messages Sent as Share', y=1.03, weight='bold')
        # configure legend
        handles, labels = m_pie.get_legend_handles_labels()
        plt.legend(handles[::-1], (m.name for m in members[::-1]), loc='center', bbox_to_anchor=(0.95, 0.5))

    def wdcount():
        """Words per message for each member as bar graph"""
        plt.subplot(223)
        wc_avg = [sum(m.words.values()) / sum(m.days) for m in members]
        barlst = plt.barh(range(len(members)), wc_avg, align='center', height=0.5)
        # set bar colors
        for i in range(len(barlst)):
            barlst[i].set_color(SPEC_THEME[i])
        plt.xlim([0, max(wc_avg)*1.15])
        plt.yticks(range(len(members)), (m.name for m in members))
        plt.gca().yaxis.grid(False)
        plt.gca().tick_params(top='off', right='off')
        # annotate bars exact value
        for i in range(len(members)):
            plt.text(wc_avg[i]+max(wc_avg)*0.02, i, format(wc_avg[i], '.3f'), ha='left', va='center')
        plt.title('Average Words per Message', y=1.03, weight='bold')

    def wdcountpie():
        """Total word count for each member as pie chart"""
        w_pie = plt.subplot(224)
        # explode max
        wc_total = [sum(m.words.values()) for m in members]
        explode = tuple(0.1 if sum(m.words.values())==max(wc_total) else 0 for m in members)
        plt.pie(wc_total, explode=explode, labels=[' ' for m in members], colors=SPEC_THEME, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Total Words Written as Share', y=1.03, weight='bold')
        # configure legend, spacing
        handles, labels = w_pie.get_legend_handles_labels()
        plt.legend(handles[::-1], (m.name for m in members[::-1]), loc='center', bbox_to_anchor=(0.95, 0.5))

    # set up general data
    msgs = [sum(m.days) for m in members]
    # plot
    msgcount()
    msgcountpie()
    wdcount()
    wdcountpie()
    # adjust spacing
    plt.subplots_adjust(wspace=0, hspace=0.40)


def key_event(e):
    """Handles key events for scrolling through the plots"""
    global curr_pos
    # Handle arrow keys
    if e.key == 'right':
        curr_pos += 1
    elif e.key == 'left':
        curr_pos -= 1
    else:
        return
    # plot next site
    curr_pos = curr_pos % len(plots)
    plt.clf()
    plots[curr_pos](members)
    fig.canvas.draw()


if __name__ == '__main__':
    chat = Chat(FILE)
    members = sorted(chat.process(), key=lambda m: sum(m.days))
    SPEC_THEME = SPEC_THEME[len(SPEC_THEME)-len(members):] if len(members) <= len(SPEC_THEME) else sample(list(clrs.cnames), len(members))
    # init key event handling vars
    curr_pos = 0
    plots = [plot_trends, plot_usetime, plot_users]
    # set custom style
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['xtick.major.pad'] = 10
    plt.rcParams['ytick.major.pad'] = 10
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['patch.linewidth'] = 0.4  # width of pie chart lines
    plt.rcParams['axes.edgecolor'] = 'black'
    # prepare window
    fig = plt.figure()
    fig.canvas.set_window_title('Whatsapp Analyzer')
    fig.canvas.mpl_connect('key_press_event', key_event)
    plots[0](members)
    # show
    plt.show()

