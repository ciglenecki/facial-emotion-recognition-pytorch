import string
import random


def bool_action(action_name):
    result = ''
    while (result != 'y') and (result != 'n'):
        result = input(action_name + "? - y/n\n")

    if result == 'y':
        result = True
    elif result == 'n':
        result = False
    return result


def number_action(number_name):
    number = ''
    while (not isinstance(number, int) and not isinstance(number, float)):
        number = input(number_name + "?\n")
        number = float(number)
    return number


def random_string(stringLength=8):
    lettersAndDigits = string.ascii_letters + string.digits
    return ''.join((random.choice(lettersAndDigits) for i in range(stringLength)))

def random_string(stringLength=8):
    lettersAndDigits = string.ascii_letters + string.digits
    return ''.join((random.choice(lettersAndDigits) for i in range(stringLength)))
