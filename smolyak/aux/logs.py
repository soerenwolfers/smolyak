'''

'''
from _io import BytesIO
import sys
import datetime

class Capturing(list):
    '''
    From http://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
    '''
    def __enter__(self):
        self._stdout,self._stderr = sys.stdout,sys.stderr
        sys.stdout, sys.stderr = self._stdio,self._errio = BytesIO(),BytesIO()
        return self
    def __exit__(self, *args):
        self.extend([self._stdio.getvalue(),self._errio.getvalue()])
        del self._stdio,self._errio    # free up some memory
        sys.stdout,sys.stderr = self._stdout,self._stderr
        
        
class Log(object):   
    def __init__(self,print_verbosity=False,write_verbosity=False,file_name=None):
        if print_verbosity is True:
            print_verbosity = lambda _: True
        if print_verbosity is False:
            print_verbosity = lambda _: False
        self.print_verbosity=print_verbosity
        self.file_name=file_name
        if write_verbosity and not self.file_name:
            raise ValueError('Specify file_name to write log in file')
        if write_verbosity is True:
            write_verbosity = lambda _: True
        if write_verbosity is False:
            write_verbosity = lambda _: False
        self.write_verbosity=write_verbosity
        self.entries=[Entry(group='',message='Create log')]
        
    def log(self,message=None,group=None,tags=None):
        self.entries.append(Entry(group=group,message=message,tags=tags))
        self.entries[-1].print_entry(self.print_verbosity)
        self.entries[-1].write_entry(self.file_name,self.write_verbosity)
        
    def __str__(self):
        #with Capturing() as output:
        #    for entry in self.entries:
        #        entry.print_entry()
        return '\n'.join([entry.__str__() for entry in self.entries])
        
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
        
    def __str__(self):
        string='<'+str(self.time)
        if self.group:
            string+= ' | '+ self.group
        string+='> '+self.message
        return string
        
    def print_entry(self,filter=lambda _: True):  # @ReservedAssignment
        if filter(self):
            print(self)    
            
    def write_entry(self,file_name,filter=lambda _: True):  # @ReservedAssignment
        if filter(self):
            with open(file_name,'a') as fp:
                fp.write(self.__str__()+'\n')
