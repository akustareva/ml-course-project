from dictionary.Constants import *

operation_type = [
    HKCR,  # HKEY_CLASSES_ROOT
    HKCU,  # HKEY_CURRENT_USER
    HKLM,  # HKEY_LOCAL_MACHINE
    HKU,  # HKEY_USERS
    HKCC,  # HKEY_CURRENT_CONFIG
    OTHER_REG_OP,  # \REGISTRY\A\*
    FASTIO_OP,
    IRP,
    TCP_OR_UDP,
    THREAD_ACTION,  # Thread Create/Exit/Profiling
    PROCESS_ACTION,  # Process Create/Exit/Profiling/Start
    QUERY_ACTION,
    READ_FILE,
    WRITE_FILE,
    CLOSE_FILE,
    CREATE_FILE,
    OTHER_OP_WITH_FILE,
    CONTROL,  # *Control
    CHANGE_DIR,
    LOAD_IMG,
    UNKNOWN,
    OTHER
]
operations_count = len(operation_type)

result_type = [
    SUCCESS,
    UNSUCCESS,
    OTHER
]
results_count = len(result_type)

path_type = [
    EXE,
    EXE_WIN,
    LIB,
    REGISTRY,
    TMP,
    EMPTY,
    OTHER
]
paths_count = len(path_type)

operation_type_map = {}
result_type_map = {}
path_type_map = {}

for i in range(len(operation_type)):
    operation_type_map[operation_type[i]] = i

for i in range(len(result_type)):
    result_type_map[result_type[i]] = i

for i in range(len(path_type)):
    path_type_map[path_type[i]] = i
