import os, psutil
import learner


class Menu():
    '''
    Class that encapsulates a general menu.
    '''

    def __init__(self):
        pass

    def run(self, opts, title=None, cast=int, print_exit=True):
        '''
        Print menu and return the choosen option.
        '''
        self.__print(opts, title=title)
        opt_id = self.__read_opt(maxOpt=len(opts), cast=cast)
        if opt_id:
            return opts[opt_id - 1]

    def __print(self, opts, cls=True, title=None):
        '''
        Format and print options of a menu.
        There is the option to clear the console,
        and to include a title for the menu.
        '''
        if cls:
            os.system('cls' if os.name == 'nt' else 'clear')
            print()
        if title:
            print('\n\n\t{}\n'.format(title))
        for idx, opt in enumerate(opts):
            print('\t{:02}. {}'.format(idx+1, opt))

    def __read_opt(self, maxOpt, cast=int):
        '''
        Read option from console. It checks if the input
        value can be casted to the required type and whether
        the value is in range or not. 
        '''
        try:
            opt_id = cast(input('\n\tPlease enter your option: '))
        except ValueError:
            err = 'Expected a {}...'.format(cast)
            self.__print_invalid(err)
            return
        print('\tYou have entered: {}'.format(opt_id))
        if opt_id not in range(1,maxOpt+1):
            err = '{} is out of range...'.format(opt_id)
            self.__print_invalid(err)
            return
        else:
            return opt_id

    def __print_invalid(self, addInfo=None):
        '''
        Print error message due to an error in the input.
        Allows to include a message with additional information.
        '''
        print('\n\tInvalid option! {}'.format(addInfo))
        input('\n\tPress Enter to continue...')

    def exit(self, prnt=False):
        '''
        Exit message.
        '''
        if type(prnt) is bool and prnt:
            print('\n\n\tBye bye!\n')
        return True


class Helper:
    
    def __init__(self):
        pass

    def read(self, text, cast=str):
        while True:
            try:
                intr = cast(input('\n\t{}: '.format(text)))
            except ValueError:
                print('\n\tExpected a {}...'.format(cast))
                continue
            return intr

    def _print(self, text):
        print('\n\t{}'.format(text))

    def _continue(self):
        input('\n\tPress Enter to continue...')

    def memory_usage(self):
        self.__cls()
        print('\n\t- Total memory: {}GB'.format(psutil.virtual_memory().total >> 30))
        print('\t- Used memory: {}GB'.format(psutil.virtual_memory().used >> 30))
        print('\t- Free memory: {}GB'.format(psutil.virtual_memory().available >> 30))
        print('\t- Used memory: {}%'.format(psutil.virtual_memory().percent))
        self._continue()

    def disk_usage(self):
        self.__cls()
        print('\n\t- Total space: {}GB'.format(psutil.disk_usage('/').total >> 30))
        print('\t- Used space: {}GB'.format(psutil.disk_usage('/').used >> 30))
        print('\t- Free space: {}GB'.format(psutil.disk_usage('/').free >> 30))
        print('\t- Used space: {}%'.format(psutil.disk_usage('/').percent))
        self._continue()

    def __cls(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print()


if __name__ == "__main__":
    # Define menu options and call function that runs the menu
    menu = Menu()
    helper = Helper()
    learner = learner.DeepLearner()
    title = 'Deep Playground ready!'
    opts = {'Train models.': learner.prepare_training,
            'Test trained models.': learner.prepare_test,
            'Update hyperparameters and optimizer.': learner.update_options,
            'Produce table with results': learner.results_table,
            'Check available memory.': helper.memory_usage,
            'Check disk usage.': helper.disk_usage,
            'Exit.': menu.exit
           }
    stop = False
    while not stop:
        opt = menu.run(list(opts.keys()), title=title)
        if opt:
            stop = opts[opt]()
    helper._print('Bye bye!\n')
