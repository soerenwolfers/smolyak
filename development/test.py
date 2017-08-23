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
import re
from multiprocessing import Process, Lock, Pool
import cPickle
MSG_MEMPROF = 'Could not store memory profiler. Install memory_profiler via pip install memory_profiler.'
MSG_SOURCE = 'Could not find source code'
MSG_SERIALIZER = ('Could not find dill. Some items might not be storable. '
    + ('Storage of numpy arrays will be slow' if sys.version[0] < 3 else '')
    + 'Install dill via pip install dill.')
MSG_STORE_RESULT = 'Could not serialize results'
MSG_STORE_INFO = lambda keys: 'Could not store keys {}.'.format(keys)
MSG_FINISH_EXPERIMENT = lambda i, runtime: 'Experiment {} finished. Runtime: {}'.format(i, runtime)
MSG_RUNTIME_SIMPLE = lambda runtime: 'Runtime: ' + str(runtime) + 's. For more detailed information use \'runtime_profile=True\''
MSG_FINISHED = 'Done'
MSG_NO_MATCH = 'Could not find matching experiment series'
MSG_MULTI_MATCH = lambda series:'Multiple matching experiment series (to iterate through all use need_unique=False):\n{}'.format('\n'.join(series))
MSG_UNUSED = 'Passed configuration dictionary is unused when running experiment series with function'
MSG_ERROR_LOAD = lambda name: 'Error loading {}'.format(name)
GRP_WARN = 'Warning'
GRP_ERR = 'Error'

def conduct(func, experiments, name=None, path='experiments', supp_data=None,
            analyze=None, runtime_profile=False, memory_profile=False,
            no_date=False, no_dill=False):
    '''   
    Call :code:`func` once for each entry of :code:`experiments` and store
    results along with auxiliary information such as runtime and memory usage.
    Each entry of experiments is passed as a whole to :code:`func`, e.g.:
            def func(experiment):
                return experiment['a']*experiment['x']**experiment['exponent']
            base={'exponent':2,'a':5}
            experiments=[dict('x'=x,**base) for x in range(10)]
            conduct(func,experiments)
    In practice, :code:`func` can be a numerical algorithm and :code:`experiments` 
    can be a list of different mesh resolutions, a list of different
    subroutines, etc.
    
    This function stores the following files and directories in a directory 
    specified by :code:`name` and :code:`path`:
        *info.pkl:
            *name: Name of experiment series (str)
            *time: Time of execution (datetime.datetime)
            *func: Parameter :code:`func`
            *experiments: Parameter :code:`experiments`
            *runtime: Runtime of each experiment (list of floats)
            *status: Status of each experiment (list of ('queued'/'finished'/'failed'))
            *(optional)supp_data: Parameter :code:`supp_data`
        *log.txt
        *results.pkl: List of results of experiments 
        *source.txt: Source code of the module containing :code:`func`
        *(optional)stderr.txt
        *For each experiment a subdirectory "experiment<i>" with:
            *user_files/ (Working directory for call of :code:`func`)
            *input.txt: Argument passed to :code:`func`
            *stderr.txt:
            *stdout.txt:
            *(optional)runtime_profile.txt: Extensive runtime information for each experiment (list of strings)
            *(optional)memory_profile.txt: Memory usage information for each experiment (list of strings)
        *(optional) analysis/: output of function :analysis:
            *stderr.txt
            *stdout.txt
            *user_files/ (Working directory for call of :code:`analyze`

        
    Both info.pkl and results.pkl are created with pickle, for technical
    reasons they contain multiple concatenated pickle streams. To load these files,
    and automatically join the contents of info.pkl into a single dictionary and
    the contents of results.pkl into a single list, use function :code:`load` 
     
    :param func: Function to be called with different experiment configurations
    :type func: function
    :param experiments: Experiment configurations
    :type experiments: Iterable
    :param name: Unique name of experiment series. Using func.__name__ if not provided
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
    '''
    if not name:
        try: 
            name = func.__name__
        except AttributeError:
            name = func.__class__.__name__
    directory = _get_directory(name, path, no_date)
    ###########################################################################
    log_file = os.path.join(directory, 'log.txt')
    results_file = os.path.join(directory, 'results.pkl')
    info_file = os.path.join(directory, 'info.pkl')
    source_file_name = os.path.join(directory, 'source.txt')

    ###########################################################################
    MSG_START = 'Starting experiment series \'{}\' with {} experiments:\n\t{}'.format(name, len(experiments), '\n\t'.join(map(str, experiments)))
    MSG_INFO = 'This log and all outputs can be found in {}'.format(directory)
    MSG_TYPE = (('# Experiment series was conducted with instance of class {}'.format(func.__class__.__name__)
               if hasattr(func, '__class__') else 
               '# Experiment series was conducted with function {}'.format(func.__name__))
              + ' in the following module: \n')
  
    ###########################################################################
    log = Log(write_verbosity=True, print_verbosity=True, file_name=log_file)
    log.log(MSG_START)
    log.log(MSG_INFO)
    info = dict()
    info['name'] = name
    info['time'] = datetime.datetime.fromtimestamp(time.time())
    if supp_data:
        info['supp_data'] = supp_data
    info['runtime'] = [None] * len(experiments)
    if memory_profile:
        try:
            import memory_profiler
        except ImportError:
            log.log(group=GRP_WARN, message=MSG_MEMPROF)
            memory_profile = False
    info['status'] = ['queued'] * len(experiments)
    try: 
        source = MSG_TYPE + ''.join(inspect.getsourcelines(sys.modules[func.__module__])[0])
    except TypeError:
        log.log(group=GRP_WARN, message=MSG_SOURCE)
    info_list = [info, {'experiments':experiments}]
    if not no_dill:
        try: 
            import dill
            serializer = dill
        except ImportError:
            serializer = pickle
            log.log(group=GRP_WARN, message=MSG_SERIALIZER)
    else:
        serializer = pickle
    def store_info():
        with open(info_file, 'wb') as fp:
            for temp in info_list:
                try:
                    serializer.dump(temp, fp)
                except (TypeError, pickle.PicklingError):
                    log.log(group=GRP_WARN, message=MSG_STORE_INFO(temp.keys()))
    def store_result(result):
        with open(results_file, 'ab') as fp:
            try:
                serializer.dump([result], fp)
            except (TypeError, pickle.PicklingError):
                log.log(group=GRP_WARN, message=MSG_STORE_RESULT)
    def _store_data(file_name, data):
        if data:
            with open(file_name, 'a') as fp:
                fp.write(data)
    store_info()
    _store_data(source_file_name,source)
    old_wd = os.getcwd()
    lock=Lock()
    pool=Pool(processes=len(experiments))
    args=((i,experiment,directory,log,func,memory_profile,
     runtime_profile,store_result,store_info) for i,experiment in enumerate(experiments))
    #for arg in args:
    #    _run_single_experiment(arg)
    A=pool.map(_run_single_experiment, args)
    #for i, experiment in enumerate(experiments):
    
    os.chdir(old_wd)
    log.log(MSG_FINISHED)
    return directory

def _init(l):
    global lock
    lock = l
    
def _run_single_experiment(arg):
    print(1)
    (i,experiment,directory,log,func,memory_profile,
     runtime_profile,store_result,store_info)=arg
    stderr_file = os.path.join(directory, 'stderr.txt')
    stderr_files = lambda i: os.path.join(directory, 'experiment{}'.format(i), 'stderr.txt')
    stdout_files = lambda i: os.path.join(directory, 'experiment{}'.format(i), 'stdout.txt')
    input_files = lambda i: os.path.join(directory, 'experiment{}'.format(i), 'input.txt')
    runtime_profile_files = lambda i:os.path.join(directory, 'experiment{}'.format(i), 'runtime_profile.txt')
    memory_profile_files = lambda i:os.path.join(directory, 'experiment{}'.format(i), 'memory_profile.txt')
    experiment_user_directories = lambda i: os.path.join(directory, 'experiment{}'.format(i), 'user_files')
    MSG_EXCEPTION_ANALYSIS='Exception during online analysis. Check {}'.format(stderr_file)
    MSG_FAILED_EXPERIMENT = lambda i:'Experiment {} not completed. Check {}'.format(i, stderr_files(i))
    MSG_EXCEPTION_EXPERIMENT = lambda i: 'Exception during execution of experiment {}. Check {}'.format(i, stderr_file)
    MSG_START_EXPERIMENT = lambda i: 'Starting experiment {} with argument:\n\t{}'.format(i, str(experiment)) 
    log.log(MSG_START_EXPERIMENT(i)) 
    runtime = None
    output = None
    memory = None
    
    if  hasattr(func, '__name__'):
        temp_func = func
    else:
        temp_func = func.__call__
    experiment_directory = experiment_user_directories(i)
    os.makedirs(experiment_directory)
    os.chdir(experiment_directory)
    try:
        if memory_profile:
            import memory_profiler
            m = StringIO()
            temp_func = memory_profiler.profile(func=temp_func, stream=m, precision=4)
        if runtime_profile:
            temp_func = add_runtime(temp_func)
        stderr_append=""
        with capture_output() as c:
            tic = timeit.default_timer()
            try:
                output = temp_func(experiment)
                #info['status'][i] = 'finished'
                status='finished'
            except Exception:
                #info['status'][i] = 'failed'
                status='failed'
                stderr_append=traceback.format_exc()
            runtime = timeit.default_timer() - tic
        if stderr_append:
            log.log(group=GRP_ERR, message=MSG_FAILED_EXPERIMENT(i))
        _store_data(stderr_files(i), c.stderr+(stderr_append))
        _store_data(stdout_files(i), c.stdout)
        _store_data(input_files(i), str(experiment))
        #info['runtime'][i] = runtime
        if runtime_profile: 
            profile, output = output
            s = BytesIO()
            ps = pstats.Stats(profile, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats()
            _store_data(runtime_profile_files(i), s.getvalue())
            s.close()
        else:
            _store_data(runtime_profile_files(i), MSG_RUNTIME_SIMPLE(runtime))
        if memory_profile:
            _store_data(memory_profile_files(i), m.getvalue())
            #info['memory'][i]=_max_mem(m.getvalue())
            memory=_max_mem(m.getvalue())
    except Exception:
        lock.acquire()
        _store_data(stderr_file, traceback.format_exc())
        lock.release()
        log.log(group=GRP_ERR, message=MSG_EXCEPTION_EXPERIMENT(i))
    #if info['status'][i] == 'finished':
    if status=='finished':
        log.log(MSG_FINISH_EXPERIMENT(i, runtime))   
    os.chdir(directory)
    lock.acquire()#TODO MAKE SURE THE ORDER OF OUTPUTS IS RECOVERABLE
    store_result(output)
    lock.release()
    del output
    gc.collect()
    store_info()
    if analyze:
        lock.acquire()
        try:
            globals()['analyze'](func=analyze,path=directory,log=log)
        except:
            _store_data(stderr_file, traceback.format_exc())
            log.log(group=GRP_ERR, message=MSG_EXCEPTION_ANALYSIS)
        lock.release()
    return (runtime,status,memory)
    
def analyze(func,search_pattern='*',path='',need_unique=False,log=None,no_dill=False):
    if not no_dill:
        try: 
            import dill
            serializer = dill
        except ImportError:
            serializer = pickle
            log.log(group=GRP_WARN, message=MSG_SERIALIZER)
    else:
        serializer = pickle
    MSG_FAILED_ANALYSIS = lambda stderr_file: 'Analysis could not be completed. Check {}'.format(stderr_file)
    MSG_STORE_ANALYSIS = lambda name: 'Could not serialize results of analysis'
    def _store_data(file_name, data):
        if data:
            with open(file_name, 'w') as fp:
                fp.write(data)
    for (info,results,directory) in load(search_pattern=search_pattern,path=path,need_unique=need_unique,info_only=False):
        analysis_directory = os.path.join(directory, 'analysis')
        shutil.rmtree(analysis_directory, ignore_errors=True)
        os.mkdir(analysis_directory)
        analysis_user_directory = os.path.join(analysis_directory, 'user_files')
        shutil.rmtree(analysis_user_directory, ignore_errors=True)
        os.mkdir(analysis_user_directory)
        analysis_stderr_file = os.path.join(analysis_directory, 'stderr.txt')
        analysis_stdout_file = os.path.join(analysis_directory, 'stdout.txt')
        analysis_output_file = os.path.join(analysis_directory, 'output.pkl')
        os.chdir(analysis_user_directory)
        output=None
        stderr_append=""
        with capture_output() as c:
            try:
                output=func(results, info)
            except Exception:
                stderr_append=traceback.format_exc()
        if stderr_append:
            if log:
                log.log(group=GRP_ERR, message=MSG_FAILED_ANALYSIS(analysis_stderr_file))
            else:
                warnings.warn(message=MSG_FAILED_ANALYSIS(analysis_stderr_file))
        _store_data(analysis_stderr_file, c.stderr+stderr_append)
        _store_data(analysis_stdout_file, c.stdout)
        if output:
            with open(analysis_output_file, 'wb') as fp:
                try:
                    serializer.dump(output, fp)
                except (TypeError, pickle.PicklingError):
                    if log:
                        log.log(group=GRP_WARN, message=MSG_STORE_ANALYSIS)
                    else:
                        warnings.warn(message=MSG_STORE_ANALYSIS)
        os.chdir(directory)          

def load(search_pattern='*', path='', info_only=False, need_unique=True):
    '''
    Load results of (possibly multiple) experiment series. 
    
    Return (generator of) tuple (info,results,directory) with the contents of 
    info.pkl and results.pkl as well as the directory of the experiment series
    
    :param search_pattern: Bash style search_pattern string(s) 
    :type search_pattern: String, e.g. search_pattern='algo*'
    :param path: Path of exact location is known (possibly only partially), relative or absolute
    :type path: String, e.g. '/home/work/2017/6/<name>' or 'work/2017/6'
    :param info_only: Only load information about experiment series, not results
    :type info_only: Boolean
    :param need_unique: Require unique identification of experiment series.
    :type need_unique: Boolean
    :return: Information about run(s) and list(s) of results
    :rtype: If need_unique=True, a single tuple (info[,results],directory),
    where `info` is a dictionary containing information regarding the experiment
    series and `results` is a list containing the results of each experiment.
    If need_unique=False, a generator of tuples (info[,results],directory) 
    '''
    deserializer = pickle
    try:
        import dill
        deserializer = dill
    except ImportError:
        warnings.warn(MSG_SERIALIZER)
    def assemble_file_contents(file_name, iterable, need_start=False, update=False):
        with open(file_name, 'r') as fp:
            output = iterable()
            for i in itertools.count():
                try:
                    to_add = deserializer.load(fp)
                except Exception as e:    
                    if i == 0 and need_start:
                        raise
                    else:
                        if isinstance(e, EOFError):
                            break
                        else:
                            warnings.warn(MSG_ERROR_LOAD(file_name))
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
            return (info, results,serie)
        else:
            return (info,serie)
    if not need_unique:
        return (get_output(serie) for serie in series)
    else:
        if len(series) == 0:
            raise ValueError(MSG_NO_MATCH)
        if len(series) > 1:
            raise ValueError(MSG_MULTI_MATCH)
        return get_output(series[0])

def _is_experiment_directory(directory):
    return os.path.isfile(os.path.join(directory, 'info.pkl'))

def _max_mem(m):
    find=re.compile('.*?(\d{1,}\.\d{4}) MiB.*')
    matches=[find.match(line) for line in m.splitlines()]
    values=[float(match.groups()[0]) for match in matches if match is not None]
    return max(values)-min(values)

def _get_directory(name, path, no_date):
    if not no_date:
        date = datetime.date.today()
        directory = os.path.join(path, 'w' + date.strftime('%W') + 'y' + str(date.year)[-2:], name)
    else:
        directory = os.path.join(path, name)
    directory = os.path.abspath(directory)
    if os.path.exists(directory) and os.listdir(directory):
        if _is_experiment_directory(directory):  # Previous series will be moved in sub v0, new series will be in sub v1
            split_path = os.path.split(directory)
            temp_rel = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
            temp_directory = os.path.join(split_path[0], '.tmp', temp_rel)
            shutil.move(directory, temp_directory)
            shutil.move(temp_directory, os.path.join(directory, 'v0'))
        candidates = [os.path.split(dir)[1] for dir in os.listdir(directory)  # @ReservedAssignment
                    if os.path.isdir(os.path.join(directory, dir))
                    and re.search('^v([0-9]|[1-9][0-9]+)$', dir)]
        if candidates:
            version = max([int(dir[dir.rindex('v') + 1:]) for dir in candidates]) + 1  # @ReservedAssignment
        else:
            version = 0
        directory = os.path.join(directory, 'v' + str(version))
    try:
        os.makedirs(directory)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
    return directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform experiment series. Store (and analyze) results.')
    parser.register('type', 'bool',
                    lambda v: v.lower() in ("yes", "true", "t", "1", "y"))
    parser.add_argument("module", type=str, action='store',
                        help='Module describing experiment series')
    parser.add_argument('experiments', type=str, action='store',
                        help='List of experiment configuration')
    parser.add_argument('-c', '--config', type=str, action='store',
                        help='Arguments for initialization',
                        default='{}')
    parser.add_argument('-n', '--name', type=str, action='store',
                        default='_')
    parser.add_argument('-a', '--analyze', type=str, action='store',
                        help='Function that performs analysis on output',
                        nargs='?', const='analyze',
                        default=None)
    parser.add_argument('--memory_profile', action='store_true')
    parser.add_argument('--runtime_profile', action='store_true')
    parser.add_argument('--no_date', action='store_true')
    parser.add_argument('--no_dill', action='store_true')
    args, unknowns = parser.parse_known_args()
    args.experiments = eval(args.experiments)
    init_dict = eval(args.config)
    module_name = args.module
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        real_module_name = '.'.join(module_name.split('.')[:-1])
        module = importlib.import_module(real_module_name)
    try:  # Suppose class is last part of given module argument
        class_or_function_name = module_name.split('.')[-1]
        cl_or_fn = getattr(module, class_or_function_name)
    except AttributeError:  # Or maybe last part but capitalized?
        class_or_function_name = class_or_function_name.title()
        cl_or_fn = getattr(module, class_or_function_name)
    if args.name == '_':
        args.name = class_or_function_name
    if inspect.isclass(cl_or_fn):
        fn = cl_or_fn(**init_dict)
    else:
        if init_dict:
            warnings.warn(MSG_UNUSED)
        fn = cl_or_fn
    if args.analyze:
        try:
            split_analyze = args.analyze.split('.')
            try:
                if len(split_analyze) > 1:  # Analyze function in different module
                    analyze_module = importlib.import_module('.'.join(split_analyze[:-1]))  
                else:
                    analyze_module = module
                analyze_fn = getattr(analyze_module, split_analyze[-1])
            except AttributeError:  # is analyze maybe a function of class instance?
                analyze_fn = getattr(fn, split_analyze[:-1])
        except: 
            analyze_fn = None
            traceback.format_exc()
            warnings.warn(MSG_ERROR_LOAD('function {}'.format(args.analyze)))   
    else:
        analyze_fn = None
    conduct(func=fn, experiments=args.experiments, name=args.name,
            supp_data=sys.argv,
            analyze=analyze_fn,
            runtime_profile=args.runtime_profile,
            memory_profile=args.memory_profile,
            no_date=args.no_date,
            no_dill=args.no_dill)
