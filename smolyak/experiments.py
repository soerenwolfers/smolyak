import timeit
import pickle
import time
import os
import errno
import datetime
from smolyak.aux import files
import shutil
from smolyak.aux.more_collections import unique
import warnings
import traceback
from smolyak.aux.decorators import add_runtime
import pstats
from io import StringIO
import itertools
import sys
import gc
from IPython.utils.capture import capture_output
from _io import BytesIO
import inspect
from smolyak.aux.logs import Log
import argparse
import importlib
import random
import string

def conduct(func, tests, name=None, path='experiments', supp_data=None,
            analyze=None,runtime_profile=False, memory_profile=False, 
            no_date=False, no_dill=False):
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
            *func: Parameter :code:`func`
            *tests: Parameter :code:`tests`
            *runtime: Runtime of each test (list of floats)
            *status: Status of each test (list of ('queued'/'finished'/'failed'))
            *(optional)supp_data: Parameter :code:`supp_data`
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
    :param name: Unique name of test series. Using func.__name__ if not provided
    :type name: String
    :param path: Root directory for storage, absolute or relative 
    :type path: String
    :param supp_data: Additional information that should be stored along with 
        the results.
    :type supp_data: Any.
    :param runtime_profile: Provide extensive runtime information. This can slow
    down the execution.
    :type runtime_profile: Boolean.
    :param memory_profile: Track memory usage. This can slow down the execution.
    type memory_profile: Boolean
    :param no_date: Do not store outputs in sub-directories grouped by calendar week.
    :type date: Boolean.
    :param no_dill: Do not use dill module. Explanation: Using pickle to store 
        numpy arrays in Python2.x is slow. Furthermore, pickle cannot serialize
        Lambda functions, or not-module level functions. As an alternative, this
        function uses dill (if available) unless this parameter is set to True.
    :type no_dill: Boolean.
    
    #TODO: Online analysis, 
    '''
    if not name:
        try: 
            name=func.__name__
        except AttributeError:
            name=func.__class__.__name__
    if not no_date:
        date = datetime.date.today()
        #directory = os.path.join(path, str(date.year), str(date.month), str(date.day), name)
        #directory = os.path.join(path,'{:02d}'.format(date.month)+'_'+str(date.year)[-2:], name)
        directory = os.path.join(path, date.strftime('%W'), name)
    else:
        directory = os.path.join(path, name)
    directory = os.path.abspath(directory)
    moved=False
    if os.path.exists(directory):
        new_directory=_archive(directory)
        if new_directory!='directory':
            moved=True
            directory=new_directory
    try:
        os.makedirs(directory)
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
    log_file=os.path.join(directory,'log.txt')
    log=Log(write_verbosity=True,print_verbosity=True, file_name=log_file)
    if moved:
        log.log('Using subdirectory {}'.format(directory))
    log.log('Starting experiment series \'{}\' with {} experiments:\n\t{}'.format(name,len(tests),'\n\t'.join(map(str,tests))))
    info_file = os.path.join(directory, 'info.pkl')
    results_file = os.path.join(directory, 'results.pkl')
    info = dict()
    info['name'] = name
    info['time'] = datetime.datetime.fromtimestamp(time.time())
    if supp_data:
        info['supp_data'] = supp_data
    info['runtime'] = [None] * len(tests)
    if memory_profile:
        try:
            import memory_profiler
        except ImportError:
            warning='Could not store memory profiler. Install memory_profiler via pip install memory_profiler.'
            log.log(group='Warning',message=warning)
            memory_profile = False
    info['status'] = ['queued'] * len(tests)
    try: 
        if hasattr(func,'__class__'):
            source='# Experiment series was conducted with instance of class {}'.format(func.__class__.__name__)
        else:
            source='# Experiment series was conducted with function {}'.format(func.__name__)
        source=source+' in the following module: \n'+''.join(inspect.getsourcelines(sys.modules[func.__module__])[0])
    except TypeError:
        log.log(group='Warning',message='Could not find source code')
    info_list = [info, {'tests':tests}]
    if not no_dill:
        try: 
            import dill
            serializer = dill
        except ImportError:
            serializer = pickle
            warning = 'Could not find dill. Some items might not be storable. '
            if sys.version[0] < 3:
                warning += 'Storage of numpy arrays will be slow. '
            warning += 'Install dill via pip install dill.'
            log.log(group='Warning',message=warning)
    else:
        serializer = pickle
    def store_info():
        with open(info_file, 'wb') as fp:
            for temp in info_list:
                try:
                    serializer.dump(temp, fp)
                except (TypeError, pickle.PicklingError):
                    warning='Could not store keys {}.'.format(temp.keys())
                    log.log(group='Warning',message=warning)
    def store_result(result):
        with open(results_file, 'ab') as fp:
            try:
                serializer.dump([result], fp)
            except (TypeError,pickle.PicklingError):
                warning='Could not store results'
                log.log(group='Warning',message=warning)
    def store_test_data(name,data,i):
        if data:
            file_name=os.path.join(directory,'test{}'.format(i),name+'.txt')
            with open(file_name,'a') as fp:
                fp.write(data)
    store_info()
    source_file_name=os.path.join(directory,'source.txt')
    with open(source_file_name,'w') as fp:
        fp.write(source)
    old_wd = os.getcwd()
    if analyze:
        analysis_directory=os.path.join(directory,'analysis')
        os.mkdir(analysis_directory)
    for i, test in enumerate(tests):
        log.log('Starting test {} with:\n\t{}'.format(i, str(test)))  
        output = None
        if  hasattr(func,'__name__'):
            temp_func = func
        else:
            temp_func = func.__call__
        test_directory = os.path.join(directory, 'test{}'.format(i),'tmp')
        os.makedirs(test_directory)
        os.chdir(test_directory)
        try:
            if memory_profile:
                m = StringIO()
                temp_func = memory_profiler.profile(func=temp_func, stream=m, precision=4)
            if runtime_profile:
                temp_func = add_runtime(temp_func)
            with capture_output() as c:
                tic = timeit.default_timer()
                output = temp_func(test)
                runtime = timeit.default_timer() - tic
            info['runtime'][i] = runtime
            info['status'][i] = 'finished'
            log.log('Test {} finished. Runtime: {}'.format(i, runtime))  
            store_test_data('input',str(test),i)
            store_test_data('stdout',c.stdout,i)
            store_test_data('stderr',c.stderr,i)
            if runtime_profile: 
                profile, output = output
                s = BytesIO()
                ps = pstats.Stats(profile, stream=s)
                ps.sort_stats('cumulative')
                ps.print_stats()
                store_test_data('runtime_profile',s.getvalue(), i)
                s.close()
            else:
                store_test_data('runtime_profile','Runtime: '+str(runtime)+'s. For more detailed information use \'runtime_profile=True\'',i)
            if memory_profile:
                store_test_data('memory_profile',m.getvalue(), i)
        except Exception:
            store_test_data('stderr',traceback.format_exc(),i)
            if info['status'][i] == 'queued':
                info['status'][i] = 'failed'
                log.log(group='Error',message='Test {} failed. Check {}'.format(i,os.path.join(directory,'test{}'.format(i),'stderr.txt')))
            else: 
                log.log(group='Error',message='Exception during execution of test {}. Check {}'.format(i,os.path.join(directory,'test{}'.format(i),'stderr.txt')))
        os.chdir(directory)
        store_result(output)
        del output
        gc.collect()
        store_info()
        if analyze:
            try:
                info_tmp,results_tmp=load(path=directory)
                os.chdir(analysis_directory)
                analyze(results_tmp,info_tmp)
                os.chdir(directory)
            except:
                file_name=os.path.join(directory,'stderr.txt')
                with open(file_name,'a') as fp:
                    fp.write(traceback.format_exc())
                log.log(group='Error',message='Online analysis failed. Check {}'.format(i,os.path.join(directory,'test{}'.format(i),'stderr.txt')))
    os.chdir(old_wd)
    log.log('Done'.format(name))
    return directory
    
def load(search_pattern='*', path='', info_only=False, need_unique=True):
    '''
    Load results of (possibly multiple) test series. 
    
    Return (list of) either tuple (info,results) of contents of info.pkl and results.pkl
    or only content of info.pkl if :code:`info_only=True`.
    
    :param search_pattern: Bash style search_pattern string(s) 
    :type search_pattern: String, e.g. search_pattern='algo*'
    :param path: Path of exact location is known (possibly only partially), relative or absolute
    :type path: String, e.g. '/home/work/2017/6/<name>' or 'work/2017/6'
    :param info_only: Only load information about test series, not results
    :type info_only: Boolean
    :return: Information about run(s) and list(s) of results
    :rtype: If need_unique=True, a single tuple (info,results),
    where `info` is a dictionary containing information regarding the experiment and 
    `results` is a list containing the results of each 
    '''
    deserializer = pickle
    try:
        import dill
        deserializer = dill
    except ImportError:
        warnings.warn('Could not find dill. Try installing dill via pip install dill')
    def assemble_file_contents(file_name, iterable, need_start=False, update=False):
        with open(file_name, 'r') as fp:
            output = iterable()
            for i in itertools.count():
                try:
                    to_add = deserializer.load(fp)
                except Exception as e:    
                    if i == 0 and need_start:
                        traceback.print_exc()
                        raise IOError('Could not read file {}.'.format(file_name))  
                    else:
                        if isinstance(e, EOFError):
                            break
                        else:
                            warnings.warn('Could not load all contents of {}'.format(file_name))
                if update:
                    output.update(to_add)
                else:
                    output += to_add
            return output
    series = []
    series.extend(files.find_directories(search_pattern, path=path))
    series.extend(files.find_directories('*/' + search_pattern, path=path))
    series = [serie for serie in series if _is_experiment_directory(serie)]
    series = unique(series)
    def get_output(serie):
        info_file_name = os.path.join(serie, 'info.pkl')
        info = assemble_file_contents(info_file_name, dict, need_start=True, update=True)
        if not info_only:
            results_file_name = os.path.join(serie, 'results.pkl')
            results = assemble_file_contents(results_file_name, list, need_start=False)
            return (info, results)
        else:
            return info
    if not need_unique:
        return [get_output(serie) for serie in series]
    else:
        if len(series) == 0:
            raise ValueError('Could not find matching test series')

        if len(series) > 1:
            raise ValueError(
                ('Multiple matching test series (to iterate through all use '
                 'need_unique=False):\n{}'.format('\n'.join(series)))
            )
        return get_output(series[0])

def _is_experiment_directory(directory):
    return os.path.isfile(os.path.join(directory, 'info.pkl'))

def _archive(directory):
    if not os.listdir(directory):#No previous usage of directory, just use it
            newname=directory
    else: 
        if _is_experiment_directory(directory):#Previous series will be moved in sub v0, new series will be in sub v1
            split_path=os.path.split(directory)
            temp_rel=''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
            temp_directory=os.path.join(split_path[0],'.tmp',temp_rel)
            shutil.move(directory,temp_directory)
            shutil.move(temp_directory,os.path.join(directory,'v0'))
            #shutil.rmtree(os.path.join(split_path[0],'tmp'))
            #newname=os.path.join(directory,'v1')
        version=0       #TODO: actually make sure to use latest version, even if there is a gap
        glob
        while os.path.exists(os.path.join(directory,'v'+str(version))):
            version+=1
        newname=os.path.join(directory,'v'+str(version))
    return newname

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Perform experiment series. Store (and analyze) results.')
    parser.register('type', 'bool',
                    lambda v: v.lower() in ("yes", "true", "t", "1", "y"))
    parser.add_argument("module",type=str,action='store',
                        help='Module describing experiment series')
    parser.add_argument('tests',type=str,action='store',
                        help='List of experiment configuration')
    parser.add_argument('-c','--config', type=str,action='store',
                        help='Arguments for initialization',
                        default='{}')
    parser.add_argument('-n','--name',type=str,action='store',
                        default='_')
    parser.add_argument('--memory_profile', action='store_true')
    parser.add_argument('--runtime_profile',action='store_true')
    parser.add_argument('--no_date',action='store_true')
    parser.add_argument('--no_dill',action='store_true')
    args, unknowns = parser.parse_known_args()
    args.tests=eval(args.tests)
    init_dict=eval(args.config)
    module_name=args.module
    module=importlib.import_module(module_name)
    try:
        class_name=module_name.split('.')[-1]
        cl=getattr(module,class_name)
    except AttributeError:
        try:
            class_name=class_name.title()
            cl=getattr(module,class_name)
        except AttributeError:
            raise ValueError('Make sure that the specific module contains a class of the same name')
    if args.name=='_':
        args.name=class_name
    ob=cl(**init_dict)#TODO: Only if this is a class, otherwise use it as function
    conduct(func=ob, tests=args.tests, name=args.name, supp_data=sys.argv, 
            runtime_profile=args.runtime_profile, 
            memory_profile=args.memory_profile, 
            no_date=args.no_date, 
            no_dill=args.no_dill)    