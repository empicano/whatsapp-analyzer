import sys
import random
import numpy as np
import matplotlib.pyplot as plt, matplotlib.colors as clrs, matplotlib.dates as mdts
import datetime as dt

from scipy.interpolate import interp1d


# exported chat file
FILE = sys.argv[1] if len(sys.argv) > 1 else 'chat.txt'
# colors used when plotting user specific information
SPEC_THEME = ['#d3d3d3', '#a9a9a9', '#588c7e', '#f2e394', '#f2ae72', '#d96459', '#8c4646']
# colors used when plotting general information
GNRL_THEME = ['#14325c', '#c9c9c9', '#5398d9', '#ff0000', '#00ff00']


class Member:
    """Represents chat participator"""

    def __init__(self, name, first, period):
        """Initializes object and sets variables to default values"""
        self.name = name
        self.wc = 0  # word count
        self.words = {}
        self.hours = [[0 for _ in range(24)] for _ in range(7)]  # messages in hour at weekday
        self.days = [0 for _ in range(period)]  # messages mapped on days
        self.first = first  # date of first message

    def add_message(self, message, day, weekday, hour):
        """Adds message data to the user object"""
        self.hours[weekday][hour] += 1
        self.days[day] += 1
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


class Chat:
    """Represents the Chat data"""

    def __init__(self, path):
        """Initializes object and reads the chat file"""
        print('Reading %s ... ' % path, end='')
        chfile = open(path, 'r')
        self.chat = chfile.readlines()
        chfile.close()
        print('DONE')

    @staticmethod
    def str_to_date(s):
        """Converts string of the format dd.mm.yy to a datetime object"""
        return dt.date(2000+int(s[6:8]), int(s[3:5]), int(s[:2]))

    @staticmethod
    def date_diff(msg1, msg2):
        """Calculates number of days that lie between two given messages"""
        return (Chat.str_to_date(msg2) - Chat.str_to_date(msg1)).days

    def _rm_newlines(self):
        """Removes newline chars from messages"""
        res = []
        prev = None
        for line in self.chat:
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
        self.chat = res

    def _convert(self):
        """Converts message format to the english format"""
        print('Converting to english format ... ', end='')
        try:
            int(self.chat[0][16:18])
        except ValueError:
            # convert from german to english format
            for i in range(len(self.chat)):
                line = self.chat[i]
                date = line[:10]
                line = line[10:]
                time = line[:line.index('.')]
                line = line[line.index('-')+1:]
                time, vrna = time.split()
                # convert time
                hour = int(time[:time.index(':')])
                hour = 0 if hour == 12 else hour
                minute = time[-3:]
                time = (format(hour, '02') if vrna == 'vorm' else str(hour+12)) + minute + ':00:'
                self.chat[i] = date + time + line
        finally:
            print('DONE')

    def process(self):
        """Orders and prepares data for plotting"""
        self._rm_newlines()
        self._convert()
        # initialize vars
        members = []
        first = self.chat[0]
        period = Chat.date_diff(first, self.chat[-1]) + 1
        # process messages
        print('Preparing data for plotting ... ', end='')
        for line in self.chat:
            try:
                date = line[:8]
                hour = int(line[10:12])
                line = line[20:]
                name = line[:line.index(':')]
                line = line[line.index(': ') + 2:]
            except ValueError:
                pass  # ignore corrupted messages
            else:
                if all(member.name != name for member in members):
                    members.append(Member(name, Chat.str_to_date(date), period))
                for member in members:
                    if member.name == name:
                        member.add_message(line, Chat.date_diff(first, date), Chat.str_to_date(date).weekday(), hour)
        print('DONE')
        return members


def plot_general(members):
    """Visualizes data concerning all users"""

    # set window title
    plt.figure().canvas.set_window_title('Whatsapp Analyzer')
    period = max([len(m.days) for m in members])
    days = [sum([m.days[i] for m in members]) for i in range(period)]

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
        lbl = mxm[0].strftime('%a, %d.%m.%Y')
        plt.annotate(lbl, xy=mxm, xytext=(30, 0), textcoords='offset points', va='center', arrowprops=dict(arrowstyle='->'))

    # plot overall message count average per hour of the day
    x = np.linspace(0, 23, num=128, endpoint=True)
    overall = [e / period for e in [sum([sum([m.hours[i][j] for i in range(7)]) for m in members]) for j in range(24)]]
    # get midweek (mon, tue, wen, thu) and weekend (fri, sat, sun) hours
    midweek = [e / (period*4/7) for e in [sum([sum([m.hours[i][j] for i in range(4)]) for m in members]) for j in range(24)]]
    weekend = [e / (period*3/7) for e in [sum([sum([m.hours[i][j] for i in range(4, 7)]) for m in members]) for j in range(24)]]
    # cubic interpolate
    f = interp1d([i for i in range(24)], overall, kind='cubic')
    g = interp1d([i for i in range(24)], midweek, kind='cubic')
    h = interp1d([i for i in range(24)], weekend, kind='cubic')
    # plot
    plt.subplot(212)
    plt.plot(x, f(x), GNRL_THEME[2], ls='-', lw=1.5)
    plt.plot(x, g(x), GNRL_THEME[3], ls='--', lw=1.5)
    plt.plot(x, h(x), GNRL_THEME[4], ls='--', lw=1.5)
    # ticks, grid, labels
    plt.xticks(np.arange(min(x), max(x)+1, 2.0))
    plt.xlim([-1, 24])
    plt.grid()
    plt.title('Average Messages per Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('#Messages')
    plt.legend(['Overall', 'Midweek [Mo,Tu,We,Th]', 'Weekend [Fr,Sa,Su]'], loc=2)

    # show plots
    plt.show()


def plot_users(members):
    """Visualizes data concerning specific users"""

    # set window title
    plt.figure().canvas.set_window_title('Whatsapp Analyzer')
    # set colors
    colors = SPEC_THEME[:len(members)] if len(members) <= len(SPEC_THEME) else random.sample(clrs.cnames, len(members))
    period = max([len(m.days) for m in members])

    # total message count for each member as bar graph
    members = sorted(members, key=lambda m: sum(m.days))
    plt.subplot(221)
    msgs = [sum(m.days) for m in members]
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
    explode = tuple([0.1 if sum(m.days)==max(msgs) else 0 for m in members])
    plt.pie(msgs, explode=explode, labels=[' ' for m in members], colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Total Messages Sent as Share')
    # configure legend
    handles, labels = m_pie.get_legend_handles_labels()
    plt.legend(handles[::-1], [m.name for m in members][::-1], loc='center', bbox_to_anchor=(0.95, 0.5))

    # words per message for each member as bar graph
    plt.subplot(223)
    wc_avg = [m.wc / sum(m.days) for m in members]
    barlst = plt.barh(range(len(members)), wc_avg, align='center', height=0.4)
    # set bar colors
    for i in range(len(barlst)):
        barlst[i].set_color(colors[i])
    plt.xlim([0, max(wc_avg)*1.15])
    plt.yticks(range(len(members)), [m.name for m in members], size='small')
    # annotate bars exact value
    for i in range(len(members)):
        plt.text(wc_avg[i]+max(wc_avg)*0.02, i, format(wc_avg[i], '.3f'), ha='left', va='center')
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


if __name__ == '__main__':
    chat = Chat(FILE)
    members = chat.process()
    plot_general(members)
    plot_users(members)

