import timeit
import pickle
import time
import os
import errno
import datetime
from smolyak.misc import files
import shutil
from smolyak.misc.collections import unique
import warnings
import traceback
from smolyak.misc.decorators import add_profile
import pstats
import StringIO
import numpy as np
import itertools
import sys
import gc

def conduct(func,tests,name=None,path='',user_data=None,runtime_profile=False,memory_profile=False,overwrite=False,date=True,no_dill=False):
    '''   
    Call :code:`func` once for each entry of :code:`tests` and store
    results along with auxiliary information such as runtime and memory usage.
    Multiple parameters need to be simulated with lists or dictionaries, e.g.
            def func(test):
                return test[0]**test[1]
            tests=[[x,2] for x in range(10)]
        or
            def func(test):
                return test['a']*test['x']**test['exponent']
            base={'exponent':2,'a':5}
            tests=[dict(x=x,**base) for x in range(10)]

    More realistically, :code:`func` can be a numerical algorithm and 
    :code:`tests` can be a list of different mesh resolutions, a list of different
    subroutines, etc.
    
    Use hierarchical, date-oriented, filesystem to store results of test series
    in folder :code:`name` that contains:
        *info.pkl:
            *name: Name of test series (str)
            *time: Time of execution (datetime.datetime)
            *func: Parameter func
            *tests: Parameter tests
            *runtime: Runtime of each test (list of floats)
            *status: Status of each test (list of ('queued'/'finished'/'failed'))
            *(optional)user_data: User-provided additional data from parameter :code:`user_data`
            *(optional)runtime_profile: Extensive runtime information for each test (list of strings)
            *(optional)memory_profile: Memory usage information for each test (list of strings)

        *results.pkl: Outputs of tests
        
        *working directories 'test<i>' for each test
        
    Both info.pkl and results.pkl are created with pickle, for technical
    reasons they contain multiple concatenated pickle streams. To load these files,
    and automatically join the contents of info.pkl into a single dictionary and
    the contents of results.pkl into a single list, use load_test_series 
     
    :param func: Function to be called with different test configurations
    :type func: function
    :param tests: Test configurations
    :type tests: Iterable
    :param name: Unique name of test series. Use func.__name__ if not provided
    :type name: String
    :param path: Root directory for storage, absolute or relative 
    :type path: String
    :param user_data: Additional information that should be stored along with 
        the results.
    :type user_data: Any.
    :param runtime_profile: Provide extensive runtime information. This can slow
    down the execution.
    :type runtime_profile: Boolean.
    :param memory_profile: Track memory usage. This can slow down the execution.
    type memory_profile: Boolean
    :param overwrite: Overwrite results of existing test series of equal name.
    :type overwrite: Boolean.
    :param date: Structure outputs in date-oriented filesystem.
    :type date: Boolean.
    :param no_dill: Do not use dill module. Explanation: Using pickle to store 
        numpy arrays in Python2.x is slow. Furthermore, pickle cannot serialize
        Lambda functions, or not-module level functions. As an alternative, this
        function uses dill (if available) unless this parameter is set to True.
    :type no_dill: Boolean.
    '''
    name = name or func.__name__
    if not no_dill:
        try: 
            import dill
            serializer=dill
        except ImportError:
            serializer=pickle
            warning='Could not find dill. Some items might not be storable. '
            if sys.version[0]<3:
                warning+='Storage of numpy arrays will be slow. '
            warning+='Install dill via pip install dill.'
            warnings.warn(warning)
    else:
        serializer=pickle
    if date:
        date=datetime.date.today()
        directory=os.path.join(path,str(date.year),str(date.month),str(date.day),name)
    else:
        directory=os.path.join(path,name)
    directory = os.path.abspath(directory)
    if os.path.exists(directory):
        if  overwrite:
            shutil.rmtree(directory)
        else:
            raise ValueError('Test series already exists')
    try:
        os.makedirs(directory)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
    info_file=os.path.join(directory,'info.pkl')
    results_file=os.path.join(directory,'results.pkl')
    info=dict()
    info['name']=name
    info['time']=datetime.datetime.fromtimestamp(time.time())
    if user_data:
        info['user_data'] = user_data
    info['runtime'] = [None]*len(tests)
    if runtime_profile:
        info['runtime_profile']  = [None]*len(tests) 
    if memory_profile:
        try:
            import memory_profiler
            info['memory_profile'] = [None]*len(tests)   
        except ImportError:
            warnings.warn('Could not store memory profiler. Install memory_profiler via pip install memory_profiler.')
            memory_profile=False
    info['status']=['queued']*len(tests)
    info_list=[info,{'func':func},{'tests':tests}]
    def store_info():
        with open(info_file, 'wb') as fp:
            for temp in info_list:
                try:
                    serializer.dump(temp, fp)
                except pickle.PicklingError:
                    warnings.warn('Could not store keys {}.'.format(temp.keys()))
    def store_result(result):
        with open(results_file,'ab') as fp:
            try:
                serializer.dump([result],fp)
            except pickle.PicklingError:
                warnings.warn('Could not store results')
    store_info()
    old_wd=os.getcwd()
    for i,test in enumerate(tests):
        output = None
        temp_func = func #Each test is func with separate function, for profiling purposes
        test_directory=os.path.join(directory,'test{}'.format(i))
        os.makedirs(test_directory)
        os.chdir(test_directory)
        try:
            if memory_profile:
                m = StringIO.StringIO()
                temp_func = memory_profiler.profile(func = temp_func,stream=m,precision=4)
            if runtime_profile:
                temp_func = add_profile(temp_func)
            tic = timeit.default_timer()
            output = temp_func(test)
            runtime = timeit.default_timer()-tic
            if runtime_profile: 
                profile,output=output
                s = StringIO.StringIO()
                ps= pstats.Stats(profile,stream=s)
                ps.sort_stats('cumulative')
                ps.print_stats()
                info['runtime_profile'][i]=s.getvalue()
                s.close()
            if memory_profile:
                info['memory_profile'][i]=m.getvalue()
                #m.close()
            info['runtime'][i]=runtime
            info['status'][i]='finished'
            print('Test {} finished ({}). Runtime: {}'.format(i,str(test),runtime))    
        except Exception:
            warnings.warn(traceback.format_exc())
            info['status'][i]='failed'
            print('Test {} failed ({}).'.format(i,str(test)))
        os.chdir(directory)
        store_result(output)
        del output
        gc.collect()
        store_info()
    os.chdir(old_wd)
    return directory
    
def load(search='*',path='',info_only=False,need_unique=True):
    '''
    Load results of (possibly multiple) test series. 
    
    Return (list of) either tuple (info,results) of contents of info.pkl and results.pkl
    or only content of info.pkl if :code:`info_only=True`.
    
    :param search: Bash style search string(s) 
    :type search: String, e.g. search='algo*'
    :param path: Path of exact location is known (possibly only partially), relative or absolute
    :type path: String, e.g. '/home/work/2017/6/<name>' or 'work/2017/6'
    :param info_only: Only load information about test series, not results
    :type info_only: Boolean
    '''
    deserializer=pickle
    try:
        import dill
        deserializer=dill
    except ImportError:
        warnings.warn('Could not find dill. Try installing dill via pip install dill')
    def assemble_file_contents(file_name,iterable,need_start=False,update=False):
        with open(file_name,'r') as fp:
            output=iterable()
            for i in itertools.count():
                try:
                    to_add=deserializer.load(fp)
                except Exception as e:    
                    if i==0 and need_start:
                        traceback.print_exc()
                        raise IOError('Could not read file {}.'.format(file_name))  
                    else:
                        if isinstance(e,EOFError):
                            break
                        else:
                            warnings.warn('Could not load all contents of {}'.format(file_name))
                if update:
                    output.update(to_add)
                else:
                    output+=to_add
            return output
    if isinstance(search, (str, unicode)):
        search=[search]
    series=[]
    for pattern in search:
        series.extend(files.find_directories(pattern,path=path))
    for pattern in search:
        series.extend(files.find_directories('*/'+pattern,path=path))
    series=[serie for serie in series if os.path.isfile(os.path.join(serie,'info.pkl'))]
    series=unique(series)
    def get_output(serie):
        info_file_name=os.path.join(serie,'info.pkl')
        info = assemble_file_contents(info_file_name,dict,need_start=True,update=True)
        if not info_only:
            results_file_name=os.path.join(serie,'results.pkl')
            results = assemble_file_contents(results_file_name,list,need_start=False)
            return (info,results)
        else:
            return info
    if not need_unique:
        return [get_output(serie) for serie in series]
    else:
        if len(series)==0:
            raise ValueError('Could not find matching test series')

        if len(series)>1:
            raise ValueError(
                ('Multiple matching test series (to iterate through all use '
                 'need_unique=False):\n{}'.format('\n'.join(series)))
            )
        return get_output(series[0])

if __name__=='__main__':
    def np_rand(test):
        X=np.random.rand(test['n'],test['n'])
        return X
    tests=[{'n':int(10**(i/2.0))} for i in range(9)]
    path=conduct(func=np_rand,tests=tests,overwrite=True,memory_profile=True)
    info,results=load()
    pass
    