import telepot
from telepot.loop import MessageLoop
from pprint import pprint

bot_id = '402613973:AAFFCgZa7Y3DUbpMVbKH8VKsAJl1GN22YGs'
user_id_list = [339364204, 517372450]

def send_message(message):
    bot = telepot.Bot(bot_id)
    for user_id in user_id_list:
        bot.sendMessage(user_id, message)
        print("Sent a message over telegram.")
