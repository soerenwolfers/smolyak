'''

'''
from cStringIO import StringIO
import sys
import datetime

class Capturing(list):
    '''
    From http://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
    '''
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout
        
class Log(object):
    class Entry(object):
        def __init__(self,group=None,message=None,tags=None):
            if group is None:
                self.group=''
            else:
                self.group=group
            if message is None:
                self.message=''
            else:
                self.message=message
            if tags is None:
                self.tags=[]
            else:
                self.tags=tags
            self.time=datetime.datetime.now()
            
        def print_entry(self,filter=lambda _: True):  # @ReservedAssignment
            if filter(self):
                print str(self.time) + ' ('+ self.group + ') ' +self.message       
            
    def __init__(self,verbosity=lambda _: False):
        if verbosity is True:
            verbosity = lambda _: True
        if verbosity is False:
            verbosity = lambda _: False
        self.verbosity=verbosity
        self.entries=[Log.Entry(group='',message='Create log')]
        
    def log(self,group=None,message=None,tags=None):
        self.entries.append(Log.Entry(group=group,message=message,tags=tags))
        self.entries[-1].print_entry(self.verbosity)
        
    def __str__(self):
        with Capturing() as output:
            for entry in self.entries:
                entry.print_entry()
        return '\n'.join(output)
        
    @staticmethod            
    def filter_generator(require_group=None,require_tags=None,require_message=None,exclude_group=None,exclude_tags=None,exclude_message=None):
        def filter(entry):  # @ReservedAssignment
            if require_group is None:
                require_group=[]
            else:
                if not type(require_group) is list:
                    require_group=[require_group]
            if require_tags is None:
                require_tags=[]
            else:
                if not type(require_tags) is list:
                    require_tags=[require_tags]
            if require_message is None:
                require_message=[]
            else:
                if not type(require_message) is list:
                    require_message=[require_message]         
            if exclude_group is None:
                exclude_group=[]
            else:
                if not type(exclude_group) is list:
                    exclude_group=[exclude_group]
            if exclude_tags is None:
                exclude_tags=[]
            else:
                if not type(exclude_tags) is list:
                    exclude_tags=[exclude_tags]
            if exclude_message is None:
                exclude_message=[]
            else:
                if not type(exclude_message) is list:
                    exclude_message=[exclude_message]
            return  (
                        (any([group == entry.group for group in require_group]) or not require_group)
                        and 
                        all([tag in entry.tags for tag in require_tags])
                        and 
                        all([message in entry.message for message in require_message])
                        and
                        not entry.group in exclude_group
                        and
                        all([tag not in entry.tags for tag in exclude_tags])
                        and
                        all([message not in entry.message for message in exclude_message])
                    )
        return filter
                
if __name__=='__main__':
    log=Log()
    log.log(message='Wait')
    log.log(message='Cleanup')
    log.log(group='3',message='whatdup')
    log.print_entries(require_message='up')
    print(log)
    