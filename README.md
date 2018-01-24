# Analyze your Whatsapp Chats
Whatsapp Analyzer is a statistical analysis tool for Whatsapp chats.

![example](example.png)

### Some of the plots
- Total messages sent for each user
- Average words per message for each user
- Messages sent as share for each user
- Words written as share for each user
- Average messages in specific hour
- Number of messages mapped on days

# How to Use
First of all you need to export your chat data as text file to your computer. To do that, select the (group or personal) chat you want to analyze on your mobile phone. Under **group / contact info** you will find the button **export chat** - choose **without media**.

The chat format in the exported text file is different for some app languages. It has to be in the english format:
```
dd.mm.yy, hh:mm:ss: Rose Marie: Hello Mary, how are you?
24.01.18, 15:03:56: Mary Jane: I'm very well, thank you. I found this funny picture: <image omitted>
```

At the top of the analyzer.py file you can change the path to the chat file. In the main function you can choose which plots you want to see.

# Development
The project is still in an early phase. Some of what's to come:

- Convert from python2 to python3
- Most used words and most important words (tf-idf) for each user
- Use of emojis
- Use of media
- Scatter plots and radar charts, if i find suitable data to display in them
