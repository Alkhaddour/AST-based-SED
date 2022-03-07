

def prompt_yes_no_msg(msg):
    print(f"{msg}(Y/N): ", end='')
    ans = input().lower()
    if ans == 'y':
        return True
    else:
        return False


def cprint(msg, verbose=True):
    if verbose:
        print(msg)


