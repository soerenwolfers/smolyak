class DefaultDict(dict):
    '''
    Dictionary that returns default value on unknown keys
    '''
    def __init__(self, default):
        '''
        :param default: Default values for unknown keys
        :type default: Function
        '''
        self.default = default
        dict.__init__(self)

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            result = self[key] = self.default(key)
            return result