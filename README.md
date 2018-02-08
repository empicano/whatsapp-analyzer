# Introduction

Whatsapp-Analyzer is a statistical analysis tool for Whatsapp chats. Using Matplotlib, it generates various plots showing, for example, the average number of words chat participants write per message. For the analysis it works on the chat files that can be exported from Whatsapp. Below you can see a selection of plots it produces when analyzing a group chat of mine:

![example](example.png)

# List of Plots

- Message Trend
- Activity Periods
- Number of messages (total and share)
- Average words per message
- Words written as share
- Average messages in/on specific hour or weekday
- Number of media files sent

# How to Use

This program runs on Linux. To get started, you need to export the chat you want to analyze to your computer. To do that, open Whatsapp on your mobile phone and select the chat. Under **group / contact info** you will find the button **export chat** - choose **without media**.

Whatsapp is weirdly inconsistent with the format of exported files. Depending on mobile phone OS and language, the time, date and status message format will be different. This program expects the following format:


```
dd.mm.yy, hh:mm:ss: Rose Marie: Darling! Its been an age! Tell me, how are you?
24.01.18, 09:03:56: Mary Jane: Simply splendid. I purchased a new estate: <image omitted>
```

Place your chat file named **_chat.txt** in the same directory as the analyzer. Alternatively you can hand over a file path as command line argument. Run `python3 analyzer.py [file path]` in your terminal to start the analysis. You can switch between plots using the arrow keys.

# Needed Dependencies

- [matplotlib](https://matplotlib.org)
- [scipy](https://www.scipy.org)
- [numpy](http://www.numpy.org)

