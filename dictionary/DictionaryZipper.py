from dictionary.Constants import *


def zip_operation(op, p):
    operation = op.upper()
    if p is not None:
        path = p.upper()
        if operation.startswith('REG'):
            if path.startswith('HKCR') or path.startswith('HKCU') or path.startswith('HKLM') or path.startswith('HKCC'):
                return path[:4]
            if path.startswith('HKU'):
                return HKU
            if path.startswith('\\REGISTRY\\A\\'):
                return OTHER_REG_OP
    if operation.startswith('FASTIO_'):
        return FASTIO_OP
    if operation.startswith('IRP'):
        return IRP
    if operation.startswith('TCP') or operation.startswith('UDP'):
        return TCP_OR_UDP
    if operation.startswith('THREAD '):
        return THREAD_ACTION
    if operation.startswith('PROCESS '):
        return PROCESS_ACTION
    if operation.startswith('QUERY'):
        return QUERY_ACTION
    if operation == 'READFILE':
        return READ_FILE
    if operation == 'WRITEFILE':
        return WRITE_FILE
    if operation == 'CLOSEFILE':
        return CLOSE_FILE
    if operation == 'CREATEFILE':
        return CREATE_FILE
    if operation.startswith('SET') and operation.endswith('FILE') or operation == 'LOCKFILE' or \
            operation == 'FLUSHBUFFERSFILE' or operation == 'CREATEFILEMAPPING' or operation.startswith('UNLOCKFILE'):
        return OTHER_OP_WITH_FILE
    if operation.endswith('CONTROL'):
        return CONTROL
    if operation == '<UNKNOWN>':
        return UNKNOWN
    if operation.endswith('CHANGEDIRECTORY'):
        return CHANGE_DIR
    if operation == 'LOAD IMAGE':
        return LOAD_IMG
    print('Unknown operation:', op)
    return 'OTHER'


def zip_path(p):
    if p is None or p == '':
        return EMPTY
    path = p.upper()
    if path.split("\\")[0] in REG_ROOT or path.startswith('\\REGISTRY\\A\\'):
        return REGISTRY
    else:
        parts = path.split(".")
        if len(parts) == 1:
            return OTHER
        elif parts[-1] == "TMP":
            return TMP
        elif parts[-1] in LIB_EXT:
            return LIB
        elif parts[-1] in EXE_EXT:
            if path.startswith("C:\\WINDOWS\\"):
                return EXE_WIN
            else:
                return EXE
        else:
            return OTHER


def zip_result(res):
    if res is None:
        return SUCCESS
    result = res.upper()
    if res == '' or result == 'NONE' or result == 'SUCCESS':
        return SUCCESS
    if result in UNSUCCESSFUL_RESULTS or result.endswith('NOT SUPPORTED'):
        return UNSUCCESS
    print('Unknown result:', res)
    return OTHER


def zip_event(parameters):
    parameters['Operation'] = zip_operation(parameters['Operation'], parameters['Path'])
    parameters['Result'] = zip_result(parameters['Result'])
    parameters['Path'] = zip_path(parameters['Path'])
