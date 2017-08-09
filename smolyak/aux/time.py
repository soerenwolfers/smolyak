'''
time.sleep() substitute 
'''
def snooze(value):
    '''
    Keep busy for some time (very roughly and depending on machine value is ms)
    :param value: Time. Actual busy time depends on machine
    :type value: Number
    '''
    for i in range(int(0.65e3 * value)):
        __ = 2 ** (i / value)
    return 0