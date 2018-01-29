import sys
import random
import numpy as np
import matplotlib.pyplot as plt, matplotlib.colors as clrs, matplotlib.dates as mdts
import datetime as dt

from scipy.interpolate import interp1d


# exported chat file
FILE = sys.argv[1] if len(sys.argv) > 1 else '_chat.txt'
# colors used when plotting user specific information
SPEC_THEME = ['#d3d3d3', '#a9a9a9', '#588c7e', '#f2e394', '#f2ae72', '#d96459', '#8c4646']
# colors used when plotting general information
GNRL_THEME = ['#14325c', '#c7c6c1', '#5398d9', '#ff0000', '#00ff00']


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


class Format:
    """Bundles Functions checking for and converting between different chat formats"""

    @staticmethod
    def lang(msg):
        """Returns chat format of message as string"""
        try:
            if len(msg) > 20 and msg[2] == msg[5] == '.' and msg[8:10] == ', ':
                int(msg[:2])
                if msg[12] == msg[15] == msg[18] == ':':
                    return 'uk'
                int(msg[10])
                int(msg[13])
                return 'ger'
            if len(msg) > 20 and msg[0] == '[':
                int(msg[1:3])
                int(msg[4:6])
                if msg[3] == msg[6] == '.':
                    if msg[9] == ',' and msg[13] == msg[16] == ':' and msg[19] == ']':
                        return 'us'
                    if msg[14] == msg[17] == ':' or msg[15] == msg[18] == ':':
                        return 'rus'
                if msg[14] == msg[17] == ':' and msg[20] == ']' and msg[3] == msg[6] == '/':
                    return 'fr'
        except ValueError:
            pass

    @staticmethod
    def convert(msg):
        """Converts message to uk english format"""
        lang = Format.lang(msg)
        if lang == 'ger':
            # convert from german format
            date = msg[:10]
            msg = msg[10:]
            time = msg[:msg.index('.')]
            msg = msg[msg.index('-')+1:]
            time, vrna = time.split()
            # convert time
            hour = int(time[:time.index(':')])
            hour = 0 if hour == 12 else hour
            minute = time[-3:]
            time = (format(hour, '02') if vrna == 'vorm' else str(hour+12)) + minute + ':00:'
            return date + time + msg
        if lang == 'us':
            # convert from us format
            return msg[1:19] + ':' + msg[20:]
        if lang == 'fr':
            # convert from french format
            return msg[1:3] + '.' + msg[4:6] + '.' + msg[9:11] + ',' + msg[11:20] + ':' + msg[21:]
        if lang == 'rus':
            # convert from russian format
            if msg.index(']') == 21:
                return msg[1:7] + msg[9:21] + ':' + msg[22:]
            else:
                return msg[1:7] + msg[9:13] + '0' + msg[13:20] + ':' + msg[21:]
        return msg


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

    def rm_newlines(self):
        """Removes newline chars from messages"""
        res = []
        prev = None
        for line in self.chat:
            if Format.lang(line):
                if prev: res.append(prev)
                prev = line
            else:
                prev = prev[:-1] + ' ' + line
        res.append(prev)
        self.chat = res

    def convert(self):
        """Converts message format to the uk english format"""
        print('Converting to english format ... ', end='')
        for i in range(len(self.chat)):
            self.chat[i] = Format.convert(self.chat[i])
        print('DONE')

    def process(self):
        """Orders and prepares data for plotting"""
        self.rm_newlines()
        self.convert()
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
    # limiters, legend, labels
    plt.ylim([0, max(days)*1.1])
    plt.xlim([dates[0]-dt.timedelta(days=len(days)*0.03), dates[len(days)-1]+dt.timedelta(days=len(days)*0.03)])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().tick_params(top='off', right='off')
    plt.legend(['Total Number on specific Day', 'Average in that Month'], loc=2)
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
    plt.plot(x, f(x), GNRL_THEME[2], ls='-', lw=2)
    plt.plot(x, g(x), GNRL_THEME[3], ls='--', lw=2)
    plt.plot(x, h(x), GNRL_THEME[4], ls='--', lw=2)
    # limiters, ticks, labels
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.xlim([-1, 24])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().tick_params(top='off', right='off')
    plt.title('Average Messages per Hour', y=1.03, weight='bold')
    plt.xlabel('Hour of the Day')
    plt.ylabel('# Messages')
    plt.legend(['Overall', 'Midweek (Mo,Tu,We,Th)', 'Weekend (Fr,Sa,Su)'], loc=2)

    # show plots
    plt.subplots_adjust(hspace=0.40)
    plt.show()


def plot_users(members):
    """Visualizes data concerning specific users"""

    # set window title
    plt.figure().canvas.set_window_title('Whatsapp Analyzer')
    # set colors
    colors = SPEC_THEME[:len(members)] if len(members) <= len(SPEC_THEME) else random.sample(list(clrs.cnames), len(members))
    period = max([len(m.days) for m in members])

    # total message count for each member as bar graph
    members = sorted(members, key=lambda m: sum(m.days))
    plt.subplot(221)
    msgs = [sum(m.days) for m in members]
    barlst = plt.barh(range(len(members)), msgs, align='center', height=0.5)
    # set bar colors
    for i in range(len(barlst)):
        barlst[i].set_color(colors[i])
    plt.xlim([0, max(msgs)*1.15])
    plt.yticks(range(len(members)), [m.name for m in members])
    plt.gca().yaxis.grid(False)
    plt.gca().tick_params(top='off', right='off')
    # annotate bars with exakt value
    for i in range(len(members)):
        plt.text(msgs[i]+max(msgs)*0.02, i, str(msgs[i]), ha='left', va='center')
    plt.title('Total Messages Sent during %d Days' % period, y=1.03, weight='bold')

    # total message count for each member as pie chart
    m_pie = plt.subplot(222)
    # explode max
    explode = tuple([0.1 if sum(m.days)==max(msgs) else 0 for m in members])
    plt.pie(msgs, explode=explode, labels=[' ' for m in members], colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Total Messages Sent as Share', y=1.03, weight='bold')
    # configure legend
    handles, labels = m_pie.get_legend_handles_labels()
    plt.legend(handles[::-1], [m.name for m in members][::-1], loc='center', bbox_to_anchor=(0.95, 0.5))

    # words per message for each member as bar graph
    plt.subplot(223)
    wc_avg = [m.wc / sum(m.days) for m in members]
    barlst = plt.barh(range(len(members)), wc_avg, align='center', height=0.5)
    # set bar colors
    for i in range(len(barlst)):
        barlst[i].set_color(colors[i])
    plt.xlim([0, max(wc_avg)*1.15])
    plt.yticks(range(len(members)), [m.name for m in members])
    plt.gca().yaxis.grid(False)
    plt.gca().tick_params(top='off', right='off')
    # annotate bars exact value
    for i in range(len(members)):
        plt.text(wc_avg[i]+max(wc_avg)*0.02, i, format(wc_avg[i], '.3f'), ha='left', va='center')
    plt.title('Average Words per Message', y=1.03, weight='bold')

    # total word count for each member as pie chart
    w_pie = plt.subplot(224)
    # explode max
    wc_total = [m.wc for m in members]
    explode = tuple([0.1 if m.wc==max(wc_total) else 0 for m in members])
    plt.pie(wc_total, explode=explode, labels=[' ' for m in members], colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Total Words Written as Share', y=1.03, weight='bold')
    # configure legend
    handles, labels = w_pie.get_legend_handles_labels()
    plt.legend(handles[::-1], [m.name for m in members][::-1], loc='center', bbox_to_anchor=(0.95, 0.5))

    # show plots
    plt.subplots_adjust(wspace=0, hspace=0.40)
    plt.show()


if __name__ == '__main__':
    chat = Chat(FILE)
    members = chat.process()
    # set style
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['xtick.major.pad'] = 10
    plt.rcParams['ytick.major.pad'] = 10
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['patch.linewidth'] = 0.4  # width of pie chart lines
    plt.rcParams['axes.edgecolor'] = 'black'
    # plot
    plot_general(members)
    plot_users(members)

